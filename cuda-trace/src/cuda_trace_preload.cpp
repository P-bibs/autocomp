#include <cuda_runtime_api.h>

#include <cxxabi.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

using CudaLaunchKernelFn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
using RegisterFatBinaryFn = void** (*)(void*);
using RegisterFatBinaryEndFn = void (*)(void**);
using RegisterFunctionFn = void (*)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);

std::mutex g_mutex;
std::unordered_map<const void*, KernelRegistration> g_kernels_by_host_fun;
std::unordered_map<std::string, KernelPtx> g_ptx_by_entry;
bool g_ptx_loaded = false;

template <typename Fn>
Fn resolve_symbol(const char* name) {
    static std::mutex resolve_mutex;
    std::lock_guard<std::mutex> lock(resolve_mutex);
    void* sym = dlsym(RTLD_NEXT, name);
    if (sym == nullptr) {
        std::fprintf(stderr, "[cuda-trace] dlsym(%s) failed: %s\n", name, dlerror());
        std::abort();
    }
    return reinterpret_cast<Fn>(sym);
}

std::string read_executable_path() {
    std::vector<char> buffer(1024);
    while (true) {
        ssize_t size = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
        if (size < 0) {
            return {};
        }
        if (static_cast<size_t>(size) < buffer.size() - 1) {
            buffer[static_cast<size_t>(size)] = '\0';
            return std::string(buffer.data());
        }
        buffer.resize(buffer.size() * 2);
    }
}

std::string run_cuobjdump_dump_ptx() {
    const std::string exe_path = read_executable_path();
    if (exe_path.empty()) {
        return {};
    }

    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) {
        std::fprintf(stderr, "[cuda-trace] pipe failed: %s\n", std::strerror(errno));
        return {};
    }

    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "[cuda-trace] fork failed: %s\n", std::strerror(errno));
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
        execlp("cuobjdump", "cuobjdump", "--dump-ptx", exe_path.c_str(), static_cast<char*>(nullptr));
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
            std::fprintf(stderr, "[cuda-trace] read failed: %s\n", std::strerror(errno));
            break;
        }
        output.append(buffer, static_cast<size_t>(nread));
    }
    close(pipe_fds[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "[cuda-trace] waitpid failed: %s\n", std::strerror(errno));
        return {};
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "[cuda-trace] cuobjdump failed for %s\n", exe_path.c_str());
        return {};
    }
    return output;
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

size_t ptx_type_size(const std::string& ptx_type) {
    if (ptx_type.size() < 2 || ptx_type[0] != '.') {
        return 0;
    }
    const std::string width = ptx_type.substr(2);
    if (width == "8") {
        return 1;
    }
    if (width == "16") {
        return 2;
    }
    if (width == "32") {
        return 4;
    }
    if (width == "64") {
        return 8;
    }
    if (width == "128") {
        return 16;
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

void parse_ptx_dump(const std::string& dump) {
    size_t pos = 0;
    while (true) {
        size_t entry_pos = dump.find(".entry", pos);
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

        size_t brace_open = dump.find('{', name_end);
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
            const std::string line = trim(dump.substr(param_pos, line_end - param_pos));
            std::istringstream iss(line);
            std::string dot_param;
            ParamSpec spec;
            iss >> dot_param >> spec.ptx_type;
            spec.size = ptx_type_size(spec.ptx_type);
            kernel_ptx.params.push_back(spec);
            param_pos = dump.find(".param", line_end);
        }

        g_ptx_by_entry[kernel_ptx.entry_name] = std::move(kernel_ptx);
        pos = scan;
    }
}

void ensure_ptx_loaded() {
    if (g_ptx_loaded) {
        return;
    }
    g_ptx_loaded = true;
    parse_ptx_dump(run_cuobjdump_dump_ptx());
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

std::string format_scalar_value(const ParamSpec& spec, const void* arg_ptr) {
    if (arg_ptr == nullptr) {
        return "<null-arg-slot>";
    }

    std::ostringstream oss;
    oss << spec.ptx_type << " raw=" << format_bytes_hex(arg_ptr, spec.size);

    if (spec.size == 1) {
        const uint8_t value = *static_cast<const uint8_t*>(arg_ptr);
        oss << " u8=" << static_cast<unsigned>(value);
    } else if (spec.size == 2) {
        uint16_t value = 0;
        std::memcpy(&value, arg_ptr, sizeof(value));
        oss << " u16=" << value;
    } else if (spec.size == 4) {
        uint32_t u32 = 0;
        float f32 = 0.0f;
        std::memcpy(&u32, arg_ptr, sizeof(u32));
        std::memcpy(&f32, arg_ptr, sizeof(f32));
        oss << " u32=" << u32 << " s32=" << static_cast<int32_t>(u32);
        if (spec.ptx_type == ".f32") {
            oss << " f32=" << f32;
        }
    } else if (spec.size == 8) {
        uint64_t u64 = 0;
        double f64 = 0.0;
        std::memcpy(&u64, arg_ptr, sizeof(u64));
        std::memcpy(&f64, arg_ptr, sizeof(f64));
        oss << " u64=" << u64 << " s64=" << static_cast<int64_t>(u64);
        if (spec.ptx_type == ".f64") {
            oss << " f64=" << f64;
        }
        if (spec.is_pointer_hint) {
            oss << " ptr=" << reinterpret_cast<const void*>(static_cast<uintptr_t>(u64));
        }
    }

    if (!spec.source_type.empty()) {
        oss << " type=\"" << spec.source_type << "\"";
    }
    return oss.str();
}

std::string resolve_ptx_entry_name(const KernelRegistration& reg) {
    if (!reg.device_fun.empty() && g_ptx_by_entry.find(reg.device_fun) != g_ptx_by_entry.end()) {
        return reg.device_fun;
    }
    if (!reg.device_name.empty() && g_ptx_by_entry.find(reg.device_name) != g_ptx_by_entry.end()) {
        return reg.device_name;
    }
    return {};
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

void log_kernel_launch(const char* api_name,
                       const void* func,
                       dim3 grid_dim,
                       dim3 block_dim,
                       void** args,
                       size_t shared_mem,
                       cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_ptx_loaded();

    const auto reg_it = g_kernels_by_host_fun.find(func);
    KernelRegistration reg;
    if (reg_it != g_kernels_by_host_fun.end()) {
        reg = reg_it->second;
    }

    const std::string entry_name = resolve_ptx_entry_name(reg);
    auto ptx_it = g_ptx_by_entry.find(entry_name);
    std::vector<ParamSpec> params;
    if (ptx_it != g_ptx_by_entry.end()) {
        params = ptx_it->second.params;
        attach_source_type_hints(reg, params);
    }

    std::fprintf(stderr,
                 "[cuda-trace] %s func=%p kernel=\"%s\" ptx_entry=\"%s\" grid=(%u,%u,%u) block=(%u,%u,%u) shared=%zu stream=%p\n",
                 api_name,
                 func,
                 reg.device_name.empty() ? "<unknown>" : reg.device_name.c_str(),
                 entry_name.empty() ? "<unknown>" : entry_name.c_str(),
                 grid_dim.x,
                 grid_dim.y,
                 grid_dim.z,
                 block_dim.x,
                 block_dim.y,
                 block_dim.z,
                 shared_mem,
                 stream);

    if (args == nullptr) {
        std::fprintf(stderr, "[cuda-trace]   args: <null>\n");
    } else if (params.empty()) {
        std::fprintf(stderr, "[cuda-trace]   args: <unknown-signature>\n");
    } else {
        for (size_t i = 0; i < params.size(); ++i) {
            const void* arg_ptr = args[i];
            std::string formatted = format_scalar_value(params[i], arg_ptr);
            std::fprintf(stderr, "[cuda-trace]   arg[%zu] @%p %s\n", i, arg_ptr, formatted.c_str());
        }
    }

    if (ptx_it != g_ptx_by_entry.end()) {
        std::fprintf(stderr, "[cuda-trace]   ptx:\n%s\n", ptx_it->second.body.c_str());
    } else {
        std::fprintf(stderr, "[cuda-trace]   ptx: <not found in embedded fatbin>\n");
    }
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
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        KernelRegistration reg;
        reg.device_fun = deviceFun == nullptr ? "" : deviceFun;
        reg.device_name = deviceName == nullptr ? "" : deviceName;
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
    log_kernel_launch("cudaLaunchKernel", func, gridDim, blockDim, args, sharedMem, stream);
    return real_fn(func, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchKernel_ptsz(const void* func,
                                              dim3 gridDim,
                                              dim3 blockDim,
                                              void** args,
                                              size_t sharedMem,
                                              cudaStream_t stream) {
    static CudaLaunchKernelFn real_fn = resolve_symbol<CudaLaunchKernelFn>("cudaLaunchKernel_ptsz");
    log_kernel_launch("cudaLaunchKernel_ptsz", func, gridDim, blockDim, args, sharedMem, stream);
    return real_fn(func, gridDim, blockDim, args, sharedMem, stream);
}
