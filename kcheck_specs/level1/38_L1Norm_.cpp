#include <algorithm>
#include <charconv>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory_resource>
#include <string>
#include <unordered_map>
#include <vector>

using Coord = std::array<long long, 2>;
using SmtString = std::pmr::string;

static inline SmtString _smt_symbol(const char* name) {
    return SmtString(name);
}
static inline void _smt_append_atom(SmtString& out, const SmtString& value) {
    out += value;
}
static inline void _smt_append_atom(SmtString& out, const char* value) {
    out += value;
}
template <typename T>
static inline void _smt_append_integral(SmtString& out, T value) {
    char buffer[32];
    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (ec == std::errc()) out.append(buffer, ptr);
}
static inline void _smt_append_atom(SmtString& out, int value) {
    _smt_append_integral(out, value);
}
static inline void _smt_append_atom(SmtString& out, long long value) {
    _smt_append_integral(out, value);
}
static inline void _smt_append_atom(SmtString& out, unsigned long long value) {
    _smt_append_integral(out, value);
}
static inline void _smt_append_atom(SmtString& out, bool value) {
    out += value ? "true" : "false";
}
template <typename T>
static inline void _smt_append_atom(SmtString& out, T value) {
    out += std::to_string(value);
}
template <typename T>
static inline std::size_t _smt_integral_size(T value) {
    char buffer[32];
    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    return ec == std::errc() ? static_cast<std::size_t>(ptr - buffer) : 0;
}
static inline std::size_t _smt_atom_size(const SmtString& value) {
    return value.size();
}
static inline std::size_t _smt_atom_size(const char* value) {
    return std::char_traits<char>::length(value);
}
static inline std::size_t _smt_atom_size(int value) {
    return _smt_integral_size(value);
}
static inline std::size_t _smt_atom_size(long long value) {
    return _smt_integral_size(value);
}
static inline std::size_t _smt_atom_size(unsigned long long value) {
    return _smt_integral_size(value);
}
static inline std::size_t _smt_atom_size(bool value) {
    return value ? 4 : 5;
}
template <typename T>
static inline std::size_t _smt_atom_size(T value) {
    return std::to_string(value).size();
}
static inline void _smt_extend_variadic(SmtString& result, const SmtString& term) {
    result.reserve(result.size() + term.size() + 2);
    result.pop_back();
    result += ' ';
    result += term;
    result += ')';
}
template <typename A, typename B>
static inline SmtString _smt_binary(const char* op, const A& a, const B& b) {
    SmtString result;
    result.reserve(std::char_traits<char>::length(op) + _smt_atom_size(a) + _smt_atom_size(b) + 4);
    result += '(';
    result += op;
    result += ' ';
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename... Args>
static inline SmtString _smt_select(const char* name, const Args&... idxs) {
    constexpr std::size_t count = sizeof...(Args);
    if constexpr (count == 0) return SmtString(name);
    std::size_t total = std::char_traits<char>::length(name) + count * 9;
    auto add_idx_size = [&](const auto& idx) { total += _smt_atom_size(idx); };
    (add_idx_size(idxs), ...);
    SmtString result;
    result.reserve(total);
    for (std::size_t i = 0; i < count; ++i) result += "(select ";
    result += name;
    auto append_idx = [&](const auto& idx) {
        result += ' ';
        _smt_append_atom(result, idx);
        result += ')';
    };
    (append_idx(idxs), ...);
    return result;
}
template <typename... Args>
static inline SmtString _smt_select_from(const SmtString& base, const Args&... idxs) {
    constexpr std::size_t count = sizeof...(Args);
    if constexpr (count == 0) return base;
    std::size_t total = base.size() + count * 9;
    auto add_idx_size = [&](const auto& idx) { total += _smt_atom_size(idx); };
    (add_idx_size(idxs), ...);
    SmtString result;
    result.reserve(total);
    for (std::size_t i = 0; i < count; ++i) result += "(select ";
    result += base;
    auto append_idx = [&](const auto& idx) {
        result += ' ';
        _smt_append_atom(result, idx);
        result += ')';
    };
    (append_idx(idxs), ...);
    return result;
}
static inline SmtString _smt_fun(const char* name, const SmtString& arg) {
    SmtString result;
    result.reserve(arg.size() + std::char_traits<char>::length(name) + 3);
    result += '(';
    result += name;
    result += ' ';
    result += arg;
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_add2(const A& a, const B& b) {
    SmtString sa;
    SmtString sb;
    _smt_append_atom(sa, a);
    _smt_append_atom(sb, b);
    if (sa == "0.0") return sb;
    if (sb == "0.0") return sa;
    return _smt_binary("+", sa, sb);
}
template <typename B>
static inline SmtString _smt_add2(SmtString a, const B& b) {
    if (a == "0.0") {
        SmtString sb;
        _smt_append_atom(sb, b);
        return sb;
    }
    SmtString sb;
    _smt_append_atom(sb, b);
    if (sb == "0.0") return a;
    if (a.size() >= 2 && a[0] == '(' && a[1] == '+') {
        _smt_extend_variadic(a, sb);
        return a;
    }
    return _smt_binary("+", a, sb);
}
template <typename A, typename B>
static inline SmtString _smt_sub2(const A& a, const B& b) {
    return _smt_binary("-", a, b);
}
template <typename A, typename B>
static inline SmtString _smt_mul2(const A& a, const B& b) {
    return _smt_binary("*", a, b);
}
template <typename A, typename B>
static inline SmtString _smt_div2(const A& a, const B& b) {
    return _smt_binary("/", a, b);
}
template <typename A, typename B>
static inline SmtString _smt_max2(const A& a, const B& b) {
    SmtString sa;
    SmtString sb;
    _smt_append_atom(sa, a);
    _smt_append_atom(sb, b);
    if (sa == "(- 1.0e309)") return sb;
    if (sb == "(- 1.0e309)") return sa;
    return _smt_binary("max", sa, sb);
}
template <typename B>
static inline SmtString _smt_max2(SmtString a, const B& b) {
    if (a == "(- 1.0e309)") {
        SmtString sb;
        _smt_append_atom(sb, b);
        return sb;
    }
    SmtString sb;
    _smt_append_atom(sb, b);
    if (sb == "(- 1.0e309)") return a;
    if (a.size() >= 4 && a[0] == '(' && a[1] == 'm' && a[2] == 'a' && a[3] == 'x') {
        _smt_extend_variadic(a, sb);
        return a;
    }
    return _smt_binary("max", a, sb);
}
template <typename A, typename B>
static inline SmtString _smt_min2(const A& a, const B& b) {
    SmtString sa;
    SmtString sb;
    _smt_append_atom(sa, a);
    _smt_append_atom(sb, b);
    if (sa == "0.0") return sb;
    if (sb == "0.0") return sa;
    return _smt_binary("min", sa, sb);
}
template <typename B>
static inline SmtString _smt_min2(SmtString a, const B& b) {
    if (a == "0.0") {
        SmtString sb;
        _smt_append_atom(sb, b);
        return sb;
    }
    SmtString sb;
    _smt_append_atom(sb, b);
    if (sb == "0.0") return a;
    if (a.size() >= 4 && a[0] == '(' && a[1] == 'm' && a[2] == 'i' && a[3] == 'n') {
        _smt_extend_variadic(a, sb);
        return a;
    }
    return _smt_binary("min", a, sb);
}
template <typename A, typename B>
static inline SmtString _smt_eq(const A& a, const B& b) {
    return _smt_binary("=", a, b);
}
template <typename A, typename B>
static inline SmtString _smt_le(const A& a, const B& b) {
    return _smt_binary("<=", a, b);
}
template <typename A, typename B>
static inline SmtString _smt_lt(const A& a, const B& b) {
    return _smt_binary("<", a, b);
}
template <typename A, typename B>
static inline SmtString _smt_mod(const A& a, const B& b) {
    return _smt_binary("mod", a, b);
}
template <typename A, typename B>
static inline SmtString _smt_int_div(const A& a, const B& b) {
    return _smt_binary("_ div", a, b);
}
static inline SmtString _smt_ite(const SmtString& cond, const SmtString& true_expr, const SmtString& false_expr) {
    SmtString result;
    result.reserve(cond.size() + true_expr.size() + false_expr.size() + 8);
    result += "(ite ";
    result += cond;
    result += ' ';
    result += true_expr;
    result += ' ';
    result += false_expr;
    result += ')';
    return result;
}
template <typename... Args>
static inline SmtString _smt_and(const Args&... conds) {
    if constexpr (sizeof...(Args) == 0) return "true";
    if constexpr (sizeof...(Args) == 1) {
        SmtString result;
        (_smt_append_atom(result, conds), ...);
        return result;
    }
    SmtString result = "(and";
    ((result += ' ', _smt_append_atom(result, conds)), ...);
    result += ")";
    return result;
}
static inline SmtString _smt_add(const std::vector<SmtString>& terms) {
    if (terms.empty()) return "0.0";
    if (terms.size() == 1) return terms.front();
    SmtString result = "(+";
    for (const auto& term : terms) result += " " + term;
    result += ")";
    return result;
}
static inline SmtString _smt_max(const std::vector<SmtString>& terms) {
    if (terms.empty()) return "(- 1.0e309)";
    SmtString acc = terms.back();
    for (std::size_t i = terms.size() - 1; i-- > 0;) {
        acc = "(ite (>= " + terms[i] + " " + acc + ") " + terms[i] + " " + acc + ")";
    }
    return acc;
}
static inline SmtString _smt_min(const std::vector<SmtString>& terms) {
    if (terms.empty()) return "0.0";
    SmtString acc = terms.back();
    for (std::size_t i = terms.size() - 1; i-- > 0;) {
        acc = "(ite (<= " + terms[i] + " " + acc + ") " + terms[i] + " " + acc + ")";
    }
    return acc;
}
template <typename F>
static inline SmtString _smt_add_range(long long n, F f) {
    if (n <= 0) return "0.0";
    std::pmr::vector<SmtString> terms;
    terms.reserve(static_cast<std::size_t>(n));
    for (long long i = 0; i < n; ++i) {
        SmtString term = f(i);
        if (term == "0.0") continue;
        terms.push_back(std::move(term));
    }
    if (terms.empty()) return SmtString("0.0");
    if (terms.size() == 1) return std::move(terms.front());
    std::size_t total = 3;
    for (const auto& term : terms) total += term.size() + 1;
    SmtString result;
    result.reserve(total);
    result += "(+";
    for (const auto& term : terms) {
        result += ' ';
        result += term;
    }
    result += ')';
    return result;
}
template <std::size_t Rank, typename F>
static inline SmtString _smt_add_nd(const std::array<long long, Rank>& bounds, F f) {
    if constexpr (Rank == 0) return f(std::array<long long, 0>{});
    for (long long bound : bounds) if (bound <= 0) return "0.0";
    std::array<long long, Rank> idx{};
    std::size_t capacity = 1;
    for (long long bound : bounds) capacity *= static_cast<std::size_t>(bound);
    std::pmr::vector<SmtString> terms;
    terms.reserve(capacity);
    while (true) {
        SmtString term = f(idx);
        if (term != "0.0") terms.push_back(std::move(term));
        std::size_t axis = Rank;
        while (axis > 0) {
            --axis;
            if (++idx[axis] < bounds[axis]) goto next_index;
            idx[axis] = 0;
        }
        break;
next_index:;
    }
    if (terms.empty()) return SmtString("0.0");
    if (terms.size() == 1) return std::move(terms.front());
    std::size_t total = 3;
    for (const auto& term : terms) total += term.size() + 1;
    SmtString result;
    result.reserve(total);
    result += "(+";
    for (const auto& term : terms) {
        result += ' ';
        result += term;
    }
    result += ')';
    return result;
}
template <typename F>
static inline SmtString _smt_max_range(long long n, F f) {
    if (n <= 0) return "(- 1.0e309)";
    SmtString result;
    bool has_term = false;
    for (long long i = 0; i < n; ++i) {
        SmtString term = f(i);
        if (term == "(- 1.0e309)") continue;
        if (!has_term) {
            result = std::move(term);
            has_term = true;
            continue;
        }
        if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'a' && result[3] == 'x') {
            _smt_extend_variadic(result, term);
        } else {
            SmtString next = "(max ";
            next.reserve(result.size() + term.size() + 8);
            next += result;
            next += ' ';
            next += term;
            next += ')';
            result.swap(next);
        }
    }
    return has_term ? result : SmtString("(- 1.0e309)");
}
template <std::size_t Rank, typename F>
static inline SmtString _smt_max_nd(const std::array<long long, Rank>& bounds, F f) {
    if constexpr (Rank == 0) return f(std::array<long long, 0>{});
    for (long long bound : bounds) if (bound <= 0) return "(- 1.0e309)";
    std::array<long long, Rank> idx{};
    SmtString result;
    bool has_term = false;
    while (true) {
        SmtString term = f(idx);
        if (term != "(- 1.0e309)") {
            if (!has_term) {
                result = std::move(term);
                has_term = true;
            } else if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'a' && result[3] == 'x') {
                _smt_extend_variadic(result, term);
            } else {
                SmtString next = "(max ";
                next.reserve(result.size() + term.size() + 8);
                next += result;
                next += ' ';
                next += term;
                next += ')';
                result.swap(next);
            }
        }
        std::size_t axis = Rank;
        while (axis > 0) {
            --axis;
            if (++idx[axis] < bounds[axis]) goto next_index;
            idx[axis] = 0;
        }
        break;
next_index:;
    }
    return has_term ? result : SmtString("(- 1.0e309)");
}
template <typename F>
static inline SmtString _smt_min_range(long long n, F f) {
    if (n <= 0) return "0.0";
    SmtString result;
    bool has_term = false;
    for (long long i = 0; i < n; ++i) {
        SmtString term = f(i);
        if (term == "0.0") continue;
        if (!has_term) {
            result = std::move(term);
            has_term = true;
            continue;
        }
        if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'i' && result[3] == 'n') {
            _smt_extend_variadic(result, term);
        } else {
            SmtString next = "(min ";
            next.reserve(result.size() + term.size() + 8);
            next += result;
            next += ' ';
            next += term;
            next += ')';
            result.swap(next);
        }
    }
    return has_term ? result : SmtString("0.0");
}
template <std::size_t Rank, typename F>
static inline SmtString _smt_min_nd(const std::array<long long, Rank>& bounds, F f) {
    if constexpr (Rank == 0) return f(std::array<long long, 0>{});
    for (long long bound : bounds) if (bound <= 0) return "0.0";
    std::array<long long, Rank> idx{};
    SmtString result;
    bool has_term = false;
    while (true) {
        SmtString term = f(idx);
        if (term != "0.0") {
            if (!has_term) {
                result = std::move(term);
                has_term = true;
            } else if (result.size() >= 4 && result[0] == '(' && result[1] == 'm' && result[2] == 'i' && result[3] == 'n') {
                _smt_extend_variadic(result, term);
            } else {
                SmtString next = "(min ";
                next.reserve(result.size() + term.size() + 8);
                next += result;
                next += ' ';
                next += term;
                next += ')';
                result.swap(next);
            }
        }
        std::size_t axis = Rank;
        while (axis > 0) {
            --axis;
            if (++idx[axis] < bounds[axis]) goto next_index;
            idx[axis] = 0;
        }
        break;
next_index:;
    }
    return has_term ? result : SmtString("0.0");
}

static inline const SmtString& _smt_runtime_cached_select_x(long long i0, long long i1, SmtString** cache) {
    const std::size_t flat = ((static_cast<std::size_t>(i0)) * 65535ULL + static_cast<std::size_t>(i1));
    if (cache[flat] != nullptr) return *cache[flat];
    cache[flat] = new SmtString(_smt_select("x", i0, i1));
    return *cache[flat];
}

static inline const SmtString& _smt_cached_prefix_x(long long i0) {
    static auto** cache = []() {
        return new SmtString*[32768ULL]();
    }();
    const std::size_t flat = static_cast<std::size_t>(i0);
    if (cache[flat] != nullptr) return *cache[flat];
    cache[flat] = new SmtString(_smt_select("x", i0));
    return *cache[flat];
}

static constexpr long long kOutputSize = 2147450880LL;
static inline bool _flat_coord_in_bounds(long long flat_coord) {
    return flat_coord >= 0 && flat_coord < kOutputSize;
}

SmtString value_at(long long flat_coord) {
    const SmtString zero = "0.0";
    const SmtString neg_inf = "(- 1.0e309)";
    if (!_flat_coord_in_bounds(flat_coord)) return _smt_symbol("invalid_flat_coord");
    Coord coord{};
    long long remaining = flat_coord;
    coord[1] = remaining % 65535LL;
    remaining /= 65535LL;
    coord[0] = remaining % 32768LL;
    auto** _smt_cache_x = new SmtString*[2147450880ULL]();

    return _smt_div2(_smt_runtime_cached_select_x(coord[0], coord[1], _smt_cache_x), _smt_div2(_smt_add_range(65535, [&](long long mean_0) -> SmtString { return _smt_fun("abs", _smt_runtime_cached_select_x(coord[0], mean_0, _smt_cache_x)); }), 65535));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "expected exactly 1 coordinate argument, got " << (argc - 1) << '\n';
        return 1;
    }
    std::pmr::unsynchronized_pool_resource smt_resource;
    std::pmr::set_default_resource(&smt_resource);
    long long flat_coord = std::strtoll(argv[1], nullptr, 10);
    if (!_flat_coord_in_bounds(flat_coord)) {
        std::cerr << "flat coordinate out of range: expected in [0, " << kOutputSize << "), got " << flat_coord << '\n';
        return 1;
    }
    const SmtString result = value_at(flat_coord);
    std::fwrite(result.data(), 1, result.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    std::_Exit(0);
}
