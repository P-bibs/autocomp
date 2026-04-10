#include <cuda_runtime_api.h>

#include <cxxabi.h>
#include <dlfcn.h>
#include <limits.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct KernelRegistration {
    std::string device_fun;
    std::string device_name;
    std::string module_path;
};

struct ParamSpec {
    std::string ptx_type;
    size_t size = 0;
    bool is_pointer_hint = false;
    std::string source_type;
};

struct KernelPtx {
    std::string entry_name;
    std::string body;
    std::vector<ParamSpec> params;
};

struct JsonSink {
    FILE* stream = nullptr;
    bool owns_stream = false;
};

using CudaLaunchKernelFn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
using RegisterFatBinaryFn = void** (*)(void*);
using RegisterFatBinaryEndFn = void (*)(void**);
using RegisterFunctionFn = void (*)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);

std::mutex g_mutex;
std::unordered_map<const void*, KernelRegistration> g_kernels_by_host_fun;
std::unordered_map<std::string, std::unordered_map<std::string, KernelPtx>> g_ptx_by_module;
std::unordered_map<std::string, bool> g_ptx_loaded_by_module;
std::unordered_map<std::string, std::string> g_ptx_load_error_by_module;
JsonSink g_json_sink;
uint64_t g_event_sequence = 0;

uint64_t monotonic_time_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

template <typename Fn>
Fn resolve_symbol(const char* name) {
    static std::mutex resolve_mutex;
    std::lock_guard<std::mutex> lock(resolve_mutex);
    void* sym = dlsym(RTLD_NEXT, name);
    if (sym == nullptr) {
        std::fprintf(stderr, "{\"event\":\"trace_error\",\"message\":\"dlsym failed\",\"symbol\":\"%s\",\"detail\":\"%s\"}\n", name, dlerror());
        std::abort();
    }
    return reinterpret_cast<Fn>(sym);
}

std::string getenv_string(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string basename_of(const std::string& path) {
    const size_t pos = path.rfind('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string trim(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && (value[start] == ' ' || value[start] == '\t' || value[start] == '\n' || value[start] == '\r')) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\n' || value[end - 1] == '\r')) {
        --end;
    }
    return value.substr(start, end - start);
}

bool should_trace_registration(const KernelRegistration& reg) {
    const std::string path_filter = getenv_string("CUDA_TRACE_FILTER_PATH_SUBSTR");
    if (!path_filter.empty() && reg.module_path.find(path_filter) == std::string::npos) {
        return false;
    }

    const std::string kernel_filter = getenv_string("CUDA_TRACE_FILTER_KERNEL_SUBSTR");
    if (!kernel_filter.empty()) {
        const bool name_match = reg.device_name.find(kernel_filter) != std::string::npos;
        const bool fun_match = reg.device_fun.find(kernel_filter) != std::string::npos;
        if (!name_match && !fun_match) {
            return false;
        }
    }

    return true;
}

std::string json_escape(const std::string& value) {
    std::ostringstream oss;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(ch));
                    oss << buffer;
                } else {
                    oss << static_cast<char>(ch);
                }
                break;
        }
    }
    return oss.str();
}

void append_json_string_field(std::ostringstream& oss, const char* key, const std::string& value, bool& first) {
    if (!first) {
        oss << ',';
    }
    first = false;
    oss << '"' << key << "\":\"" << json_escape(value) << '"';
}

void append_json_cstr_field(std::ostringstream& oss, const char* key, const char* value, bool& first) {
    append_json_string_field(oss, key, value == nullptr ? std::string() : std::string(value), first);
}

template <typename T>
void append_json_integer_field(std::ostringstream& oss, const char* key, T value, bool& first) {
    if (!first) {
        oss << ',';
    }
    first = false;
    oss << '"' << key << "\":" << value;
}

void append_json_bool_field(std::ostringstream& oss, const char* key, bool value, bool& first) {
    if (!first) {
        oss << ',';
    }
    first = false;
    oss << '"' << key << "\":" << (value ? "true" : "false");
}

void append_json_dim3_field(std::ostringstream& oss, const char* key, dim3 value, bool& first) {
    if (!first) {
        oss << ',';
    }
    first = false;
    oss << '"' << key << "\":{"
        << "\"x\":" << value.x << ','
        << "\"y\":" << value.y << ','
        << "\"z\":" << value.z
        << '}';
}

std::string pointer_string(const void* ptr) {
    std::ostringstream oss;
    oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
    return oss.str();
}

JsonSink& get_json_sink() {
    if (g_json_sink.stream != nullptr) {
        return g_json_sink;
    }

    const std::string path = getenv_string("CUDA_TRACE_JSON_PATH");
    if (path.empty()) {
        g_json_sink.stream = stderr;
        g_json_sink.owns_stream = false;
        return g_json_sink;
    }

    g_json_sink.stream = std::fopen(path.c_str(), "a");
    if (g_json_sink.stream == nullptr) {
        std::fprintf(stderr,
                     "{\"event\":\"trace_error\",\"message\":\"failed to open json output\",\"path\":\"%s\",\"errno\":%d}\n",
                     path.c_str(),
                     errno);
        g_json_sink.stream = stderr;
        g_json_sink.owns_stream = false;
    } else {
        g_json_sink.owns_stream = true;
    }
    return g_json_sink;
}

void write_json_record(const std::string& line) {
    JsonSink& sink = get_json_sink();
    std::fwrite(line.data(), 1, line.size(), sink.stream);
    std::fwrite("\n", 1, 1, sink.stream);
    std::fflush(sink.stream);
}

std::string run_cuobjdump_dump_ptx(const std::string& module_path) {
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) {
        return {};
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return {};
    }

    if (pid == 0) {
        dup2(pipe_fds[1], STDOUT_FILENO);
        dup2(pipe_fds[1], STDERR_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        unsetenv("LD_PRELOAD");
        execlp("cuobjdump", "cuobjdump", "--dump-ptx", module_path.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    close(pipe_fds[1]);
    std::string output;
    char buffer[4096];
    while (true) {
        const ssize_t nread = read(pipe_fds[0], buffer, sizeof(buffer));
        if (nread == 0) {
            break;
        }
        if (nread < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        output.append(buffer, static_cast<size_t>(nread));
    }
    close(pipe_fds[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return {};
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return {};
    }
    return output;
}

size_t parse_array_width_bytes(const std::string& token) {
    const size_t open = token.find('[');
    const size_t close = token.find(']');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        return 0;
    }
    const std::string count = token.substr(open + 1, close - open - 1);
    return static_cast<size_t>(std::strtoull(count.c_str(), nullptr, 10));
}

size_t ptx_type_size(const std::string& ptx_type) {
    if (ptx_type.size() < 2 || ptx_type[0] != '.') {
        return 0;
    }
    if (ptx_type == ".f16" || ptx_type == ".b16" || ptx_type == ".u16" || ptx_type == ".s16") {
        return 2;
    }
    if (ptx_type == ".f32" || ptx_type == ".b32" || ptx_type == ".u32" || ptx_type == ".s32") {
        return 4;
    }
    if (ptx_type == ".f64" || ptx_type == ".b64" || ptx_type == ".u64" || ptx_type == ".s64") {
        return 8;
    }
    if (ptx_type == ".b8" || ptx_type == ".u8" || ptx_type == ".s8") {
        return 1;
    }
    return 0;
}

std::vector<std::string> split_top_level_args(const std::string& signature_args) {
    std::vector<std::string> result;
    std::string current;
    int angle_depth = 0;
    int paren_depth = 0;
    int bracket_depth = 0;
    for (char ch : signature_args) {
        if (ch == '<') {
            ++angle_depth;
        } else if (ch == '>') {
            if (angle_depth > 0) {
                --angle_depth;
            }
        } else if (ch == '(') {
            ++paren_depth;
        } else if (ch == ')') {
            if (paren_depth > 0) {
                --paren_depth;
            }
        } else if (ch == '[') {
            ++bracket_depth;
        } else if (ch == ']') {
            if (bracket_depth > 0) {
                --bracket_depth;
            }
        }

        if (ch == ',' && angle_depth == 0 && paren_depth == 0 && bracket_depth == 0) {
            result.push_back(trim(current));
            current.clear();
            continue;
        }

        current.push_back(ch);
    }
    if (!trim(current).empty()) {
        result.push_back(trim(current));
    }
    return result;
}

std::string demangle_name(const std::string& name) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
    if (status != 0 || demangled == nullptr) {
        std::free(demangled);
        return name;
    }
    std::string result(demangled);
    std::free(demangled);
    return result;
}

std::vector<std::string> infer_source_arg_types(const KernelRegistration& reg) {
    const std::string candidate = reg.device_fun.rfind("_Z", 0) == 0 ? demangle_name(reg.device_fun) : reg.device_name;
    const size_t open = candidate.find('(');
    const size_t close = candidate.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        return {};
    }
    return split_top_level_args(candidate.substr(open + 1, close - open - 1));
}

ParamSpec parse_param_line(const std::string& line) {
    ParamSpec spec;
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == ".align") {
            ++i;
            continue;
        }
        if (!tokens[i].empty() && tokens[i][0] == '.') {
            spec.ptx_type = tokens[i];
            if (i + 1 < tokens.size() && tokens[i + 1].find('[') != std::string::npos) {
                spec.size = parse_array_width_bytes(tokens[i + 1]);
            } else {
                spec.size = ptx_type_size(spec.ptx_type);
            }
            break;
        }
    }
    return spec;
}

void parse_ptx_dump(const std::string& dump, std::unordered_map<std::string, KernelPtx>& out) {
    size_t pos = 0;
    while (true) {
        const size_t entry_pos = dump.find(".entry", pos);
        if (entry_pos == std::string::npos) {
            break;
        }

        size_t name_start = entry_pos + std::strlen(".entry");
        while (name_start < dump.size() && (dump[name_start] == ' ' || dump[name_start] == '\t')) {
            ++name_start;
        }
        size_t name_end = name_start;
        while (name_end < dump.size() && dump[name_end] != '(' && dump[name_end] != ' ' && dump[name_end] != '\t' && dump[name_end] != '\n') {
            ++name_end;
        }
        const std::string entry_name = dump.substr(name_start, name_end - name_start);

        size_t block_start = dump.rfind('\n', entry_pos);
        block_start = block_start == std::string::npos ? entry_pos : block_start + 1;

        const size_t brace_open = dump.find('{', name_end);
        if (brace_open == std::string::npos) {
            pos = name_end;
            continue;
        }

        int brace_depth = 1;
        size_t scan = brace_open + 1;
        while (scan < dump.size() && brace_depth > 0) {
            if (dump[scan] == '{') {
                ++brace_depth;
            } else if (dump[scan] == '}') {
                --brace_depth;
            }
            ++scan;
        }
        if (brace_depth != 0) {
            pos = brace_open + 1;
            continue;
        }

        KernelPtx kernel_ptx;
        kernel_ptx.entry_name = entry_name;
        size_t module_start = dump.rfind(".version", entry_pos);
        if (module_start != std::string::npos) {
            const size_t module_line_start = dump.rfind('\n', module_start);
            module_start = module_line_start == std::string::npos ? module_start : module_line_start + 1;
            kernel_ptx.body = trim(dump.substr(module_start));
        } else {
            kernel_ptx.body = trim(dump.substr(block_start, scan - block_start));
        }

        size_t param_pos = dump.find(".param", name_end);
        while (param_pos != std::string::npos && param_pos < brace_open) {
            size_t line_end = dump.find('\n', param_pos);
            if (line_end == std::string::npos) {
                line_end = brace_open;
            }
            kernel_ptx.params.push_back(parse_param_line(trim(dump.substr(param_pos, line_end - param_pos))));
            param_pos = dump.find(".param", line_end);
        }

        out[kernel_ptx.entry_name] = std::move(kernel_ptx);
        pos = scan;
    }
}

void ensure_ptx_loaded_for_module(const std::string& module_path) {
    if (module_path.empty()) {
        return;
    }
    if (g_ptx_loaded_by_module[module_path]) {
        return;
    }
    g_ptx_loaded_by_module[module_path] = true;
    std::string dump = run_cuobjdump_dump_ptx(module_path);
    if (dump.empty()) {
        g_ptx_load_error_by_module[module_path] = "cuobjdump_dump_ptx_failed";
        return;
    }
    parse_ptx_dump(dump, g_ptx_by_module[module_path]);
    if (g_ptx_by_module[module_path].empty()) {
        g_ptx_load_error_by_module[module_path] = "ptx_parse_found_no_entries";
        return;
    }
    g_ptx_load_error_by_module.erase(module_path);
}

void attach_source_type_hints(const KernelRegistration& reg, std::vector<ParamSpec>& params) {
    const std::vector<std::string> source_types = infer_source_arg_types(reg);
    const size_t limit = source_types.size() < params.size() ? source_types.size() : params.size();
    for (size_t i = 0; i < limit; ++i) {
        params[i].source_type = source_types[i];
        if (source_types[i].find('*') != std::string::npos) {
            params[i].is_pointer_hint = true;
        }
    }
}

const KernelPtx* find_kernel_ptx(const KernelRegistration& reg) {
    const auto module_it = g_ptx_by_module.find(reg.module_path);
    if (module_it == g_ptx_by_module.end()) {
        return nullptr;
    }
    std::vector<std::string> candidates;
    if (!reg.device_fun.empty()) {
        candidates.push_back(reg.device_fun);
        const std::string demangled = demangle_name(reg.device_fun);
        if (demangled != reg.device_fun) {
            candidates.push_back(demangled);
            const size_t open = demangled.find('(');
            if (open != std::string::npos && open > 0) {
                candidates.push_back(demangled.substr(0, open));
            }
        }
    }
    if (!reg.device_name.empty()) {
        candidates.push_back(reg.device_name);
        const std::string demangled = demangle_name(reg.device_name);
        if (demangled != reg.device_name) {
            candidates.push_back(demangled);
            const size_t open = demangled.find('(');
            if (open != std::string::npos && open > 0) {
                candidates.push_back(demangled.substr(0, open));
            }
        }
    }
    for (const std::string& candidate : candidates) {
        const auto it = module_it->second.find(candidate);
        if (it != module_it->second.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

std::string format_bytes_hex(const void* ptr, size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(ptr);
    std::ostringstream oss;
    oss << "0x";
    for (size_t i = 0; i < size; ++i) {
        static const char* hex = "0123456789abcdef";
        const unsigned char value = bytes[size - i - 1];
        oss << hex[(value >> 4) & 0xf] << hex[value & 0xf];
    }
    return oss.str();
}

void append_arg_json(std::ostringstream& oss, size_t index, const ParamSpec& spec, const void* arg_ptr) {
    oss << '{';
    bool first = true;
    append_json_integer_field(oss, "index", index, first);
    append_json_string_field(oss, "ptx_type", spec.ptx_type, first);
    append_json_integer_field(oss, "size", spec.size, first);
    append_json_string_field(oss, "slot_address", pointer_string(arg_ptr), first);
    if (!spec.source_type.empty()) {
        append_json_string_field(oss, "source_type", spec.source_type, first);
    }
    append_json_bool_field(oss, "pointer_hint", spec.is_pointer_hint, first);

    if (arg_ptr == nullptr || spec.size == 0) {
        append_json_bool_field(oss, "value_available", false, first);
        oss << '}';
        return;
    }

    append_json_bool_field(oss, "value_available", true, first);
    append_json_string_field(oss, "raw_hex", format_bytes_hex(arg_ptr, spec.size), first);

    if (spec.size == 1) {
        const uint8_t value = *static_cast<const uint8_t*>(arg_ptr);
        append_json_integer_field(oss, "u8", static_cast<unsigned>(value), first);
    } else if (spec.size == 2) {
        uint16_t value = 0;
        std::memcpy(&value, arg_ptr, sizeof(value));
        append_json_integer_field(oss, "u16", value, first);
    } else if (spec.size == 4) {
        uint32_t u32 = 0;
        float f32 = 0.0f;
        std::memcpy(&u32, arg_ptr, sizeof(u32));
        std::memcpy(&f32, arg_ptr, sizeof(f32));
        append_json_integer_field(oss, "u32", u32, first);
        append_json_integer_field(oss, "s32", static_cast<int32_t>(u32), first);
        if (spec.ptx_type == ".f32") {
            if (!first) {
                oss << ',';
            }
            first = false;
            oss << "\"f32\":" << f32;
        }
    } else if (spec.size == 8) {
        uint64_t u64 = 0;
        double f64 = 0.0;
        std::memcpy(&u64, arg_ptr, sizeof(u64));
        std::memcpy(&f64, arg_ptr, sizeof(f64));
        append_json_integer_field(oss, "u64", u64, first);
        append_json_integer_field(oss, "s64", static_cast<int64_t>(u64), first);
        if (spec.ptx_type == ".f64") {
            if (!first) {
                oss << ',';
            }
            first = false;
            oss << "\"f64\":" << f64;
        }
        append_json_string_field(oss, "pointer_value", pointer_string(reinterpret_cast<const void*>(static_cast<uintptr_t>(u64))), first);
    }

    oss << '}';
}

void append_args_json(std::ostringstream& oss, void** args, const std::vector<ParamSpec>& params) {
    oss << "\"args\":[";
    if (args != nullptr) {
        for (size_t i = 0; i < params.size(); ++i) {
            if (i != 0) {
                oss << ',';
            }
            append_arg_json(oss, i, params[i], args[i]);
        }
    }
    oss << ']';
}

std::string module_path_for_symbol(const void* symbol) {
    Dl_info info;
    if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    char resolved[PATH_MAX];
    if (realpath(info.dli_fname, resolved) == nullptr) {
        return std::string(info.dli_fname);
    }
    return std::string(resolved);
}

void log_kernel_launch(const char* api_name,
                       const void* func,
                       dim3 grid_dim,
                       dim3 block_dim,
                       void** args,
                       size_t shared_mem,
                       cudaStream_t stream) {
    const auto reg_it = g_kernels_by_host_fun.find(func);
    if (reg_it == g_kernels_by_host_fun.end()) {
        return;
    }

    const KernelRegistration& reg = reg_it->second;
    if (!should_trace_registration(reg)) {
        return;
    }

    ensure_ptx_loaded_for_module(reg.module_path);
    const KernelPtx* ptx = find_kernel_ptx(reg);
    std::vector<ParamSpec> params;
    bool args_known = false;
    std::string args_known_reason;
    if (args == nullptr) {
        args_known_reason = "launch_args_unavailable";
    } else if (reg.module_path.empty()) {
        args_known_reason = "module_path_unavailable";
    } else if (ptx == nullptr) {
        const auto error_it = g_ptx_load_error_by_module.find(reg.module_path);
        if (error_it != g_ptx_load_error_by_module.end() && !error_it->second.empty()) {
            args_known_reason = error_it->second;
        } else {
            args_known_reason = "ptx_symbol_lookup_failed";
        }
    } else {
        params = ptx->params;
        attach_source_type_hints(reg, params);
        if (params.empty()) {
            args_known_reason = "ptx_param_metadata_missing";
        } else {
            args_known = true;
        }
    }

    std::ostringstream oss;
    oss << '{';
    bool first = true;
    append_json_string_field(oss, "event", "kernel_launch", first);
    append_json_integer_field(oss, "sequence", ++g_event_sequence, first);
    append_json_integer_field(oss, "pid", static_cast<long long>(getpid()), first);
    append_json_integer_field(oss, "monotonic_ns", monotonic_time_ns(), first);
    append_json_string_field(oss, "api", api_name, first);
    append_json_string_field(oss, "module_path", reg.module_path, first);
    append_json_string_field(oss, "module_basename", basename_of(reg.module_path), first);
    append_json_string_field(oss, "kernel_name", reg.device_name, first);
    append_json_string_field(oss, "device_fun", reg.device_fun, first);
    append_json_string_field(oss, "func_ptr", pointer_string(func), first);
    append_json_string_field(oss, "ptx_entry", ptx == nullptr ? std::string() : ptx->entry_name, first);
    append_json_dim3_field(oss, "grid", grid_dim, first);
    append_json_dim3_field(oss, "block", block_dim, first);
    append_json_integer_field(oss, "shared_mem_bytes", shared_mem, first);
    append_json_string_field(oss, "stream", pointer_string(stream), first);
    append_json_bool_field(oss, "args_known", args_known, first);
    append_json_string_field(oss, "args_known_reason", args_known ? std::string() : args_known_reason, first);
    if (!first) {
        oss << ',';
    }
    first = false;
    append_args_json(oss, args, params);
    append_json_string_field(oss, "ptx", ptx == nullptr ? std::string() : ptx->body, first);
    oss << '}';

    write_json_record(oss.str());
}

}  // namespace

extern "C" void** __cudaRegisterFatBinary(void* fatCubin) {
    static RegisterFatBinaryFn real_fn = resolve_symbol<RegisterFatBinaryFn>("__cudaRegisterFatBinary");
    return real_fn(fatCubin);
}

extern "C" void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
    static RegisterFatBinaryEndFn real_fn = resolve_symbol<RegisterFatBinaryEndFn>("__cudaRegisterFatBinaryEnd");
    real_fn(fatCubinHandle);
}

extern "C" void __cudaRegisterFunction(void** fatCubinHandle,
                                        const char* hostFun,
                                        char* deviceFun,
                                        const char* deviceName,
                                        int thread_limit,
                                        uint3* tid,
                                        uint3* bid,
                                        dim3* bDim,
                                        dim3* gDim,
                                        int* wSize) {
    static RegisterFunctionFn real_fn = resolve_symbol<RegisterFunctionFn>("__cudaRegisterFunction");
    KernelRegistration reg;
    reg.device_fun = deviceFun == nullptr ? "" : deviceFun;
    reg.device_name = deviceName == nullptr ? "" : deviceName;
    reg.module_path = module_path_for_symbol(hostFun);

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_kernels_by_host_fun[hostFun] = std::move(reg);
    }

    real_fn(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

extern "C" cudaError_t cudaLaunchKernel(const void* func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void** args,
                                         size_t sharedMem,
                                         cudaStream_t stream) {
    static CudaLaunchKernelFn real_fn = resolve_symbol<CudaLaunchKernelFn>("cudaLaunchKernel");
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        log_kernel_launch("cudaLaunchKernel", func, gridDim, blockDim, args, sharedMem, stream);
    }
    return real_fn(func, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchKernel_ptsz(const void* func,
                                              dim3 gridDim,
                                              dim3 blockDim,
                                              void** args,
                                              size_t sharedMem,
                                              cudaStream_t stream) {
    static CudaLaunchKernelFn real_fn = resolve_symbol<CudaLaunchKernelFn>("cudaLaunchKernel_ptsz");
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        log_kernel_launch("cudaLaunchKernel_ptsz", func, gridDim, blockDim, args, sharedMem, stream);
    }
    return real_fn(func, gridDim, blockDim, args, sharedMem, stream);
}
