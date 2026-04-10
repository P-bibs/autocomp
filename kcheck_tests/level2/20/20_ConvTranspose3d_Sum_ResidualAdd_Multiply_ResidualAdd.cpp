#include <algorithm>
#include <charconv>
#include <array>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory_resource>
#include <string>
#include <vector>

using Coord = std::array<long long, 5>;
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
template <typename... Args>
static inline SmtString _smt_select(const char* name, const Args&... idxs) {
    constexpr std::size_t count = sizeof...(Args);
    if constexpr (count == 0) return SmtString(name);
    SmtString result;
    result.reserve(std::char_traits<char>::length(name) + count * 16);
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
    SmtString result = "(+ ";
    result += sa;
    result += ' ';
    result += sb;
    result += ')';
    return result;
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
        a.pop_back();
        a += ' ';
        a += sb;
        a += ')';
        return a;
    }
    SmtString result;
    result.reserve(a.size() + sb.size() + 5);
    result += "(+ ";
    result += a;
    result += ' ';
    result += sb;
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_sub2(const A& a, const B& b) {
    SmtString result = "(- ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_mul2(const A& a, const B& b) {
    SmtString result = "(* ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_div2(const A& a, const B& b) {
    SmtString result = "(/ ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_max2(const A& a, const B& b) {
    SmtString sa;
    SmtString sb;
    _smt_append_atom(sa, a);
    _smt_append_atom(sb, b);
    if (sa == "(- 1.0e309)") return sb;
    if (sb == "(- 1.0e309)") return sa;
    SmtString result = "(max ";
    result += sa;
    result += ' ';
    result += sb;
    result += ')';
    return result;
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
        a.pop_back();
        a += ' ';
        a += sb;
        a += ')';
        return a;
    }
    SmtString result;
    result.reserve(a.size() + sb.size() + 8);
    result += "(max ";
    result += a;
    result += ' ';
    result += sb;
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_min2(const A& a, const B& b) {
    SmtString sa;
    SmtString sb;
    _smt_append_atom(sa, a);
    _smt_append_atom(sb, b);
    if (sa == "0.0") return sb;
    if (sb == "0.0") return sa;
    SmtString result = "(min ";
    result += sa;
    result += ' ';
    result += sb;
    result += ')';
    return result;
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
        a.pop_back();
        a += ' ';
        a += sb;
        a += ')';
        return a;
    }
    SmtString result;
    result.reserve(a.size() + sb.size() + 8);
    result += "(min ";
    result += a;
    result += ' ';
    result += sb;
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_eq(const A& a, const B& b) {
    SmtString result = "(= ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_le(const A& a, const B& b) {
    SmtString result = "(<= ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_lt(const A& a, const B& b) {
    SmtString result = "(< ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_mod(const A& a, const B& b) {
    SmtString result = "(mod ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
}
template <typename A, typename B>
static inline SmtString _smt_int_div(const A& a, const B& b) {
    SmtString result = "(_ div ";
    _smt_append_atom(result, a);
    result += ' ';
    _smt_append_atom(result, b);
    result += ')';
    return result;
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
            result.pop_back();
            result += ' ';
            result += term;
            result += ')';
        } else {
            SmtString next = "(max ";
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
                result.pop_back();
                result += ' ';
                result += term;
                result += ')';
            } else {
                SmtString next = "(max ";
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
            result.pop_back();
            result += ' ';
            result += term;
            result += ')';
        } else {
            SmtString next = "(min ";
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
                result.pop_back();
                result += ' ';
                result += term;
                result += ')';
            } else {
                SmtString next = "(min ";
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

static constexpr long long kOutputSize = 134217728LL;
static inline bool _flat_coord_in_bounds(long long flat_coord) {
    return flat_coord >= 0 && flat_coord < kOutputSize;
}

SmtString value_at(long long flat_coord) {
    const SmtString zero = "0.0";
    const SmtString neg_inf = "(- 1.0e309)";
    if (!_flat_coord_in_bounds(flat_coord)) return _smt_symbol("invalid_flat_coord");
    Coord coord{};
    long long remaining = flat_coord;
    coord[4] = remaining % 64LL;
    remaining /= 64LL;
    coord[3] = remaining % 64LL;
    remaining /= 64LL;
    coord[2] = remaining % 32LL;
    remaining /= 32LL;
    coord[1] = remaining % 64LL;
    remaining /= 64LL;
    coord[0] = remaining % 16LL;

    const auto v1 = [&](long long c0, long long c1, long long c2, long long c3, long long c4) -> SmtString { return _smt_add2(_smt_add_nd<4>({32, 3, 3, 3}, [&](const auto& idx) -> SmtString { return (((((((((c2 - idx[1]) + 1)) % (2))) == (0)) && (((0) <= (((((c2 - idx[1]) + 1)) / (2)))) && ((((((c2 - idx[1]) + 1)) / (2))) < (16))) && ((((((c3 - idx[2]) + 1)) % (2))) == (0)) && (((0) <= (((((c3 - idx[2]) + 1)) / (2)))) && ((((((c3 - idx[2]) + 1)) / (2))) < (32))) && ((((((c4 - idx[3]) + 1)) % (2))) == (0)) && (((0) <= (((((c4 - idx[3]) + 1)) / (2)))) && ((((((c4 - idx[3]) + 1)) / (2))) < (32))))) ? (_smt_mul2(_smt_select("conv_transpose_weight", idx[0], c1, idx[1], idx[2], idx[3]), _smt_select("x", c0, idx[0], ((((c2 - idx[1]) + 1)) / (2)), ((((c3 - idx[2]) + 1)) / (2)), ((((c4 - idx[3]) + 1)) / (2))))) : (zero)); }), _smt_select("conv_transpose_bias", c1)); };
    const auto v2 = [&](long long c0, long long c1, long long c2, long long c3, long long c4) -> SmtString { return _smt_add2(v1(c0, c1, c2, c3, c4), _smt_select("bias", c1, 0, 0, 0)); };
    const auto v3 = [&](long long c0, long long c1, long long c2, long long c3, long long c4) -> SmtString { return _smt_add2(v2(c0, c1, c2, c3, c4), v1(c0, c1, c2, c3, c4)); };
    const auto v4 = [&](long long c0, long long c1, long long c2, long long c3, long long c4) -> SmtString { return _smt_mul2(v3(c0, c1, c2, c3, c4), v1(c0, c1, c2, c3, c4)); };
    const auto v5 = [&](long long c0, long long c1, long long c2, long long c3, long long c4) -> SmtString { return _smt_add2(v4(c0, c1, c2, c3, c4), v1(c0, c1, c2, c3, c4)); };
    return v5(coord[0], coord[1], coord[2], coord[3], coord[4]);
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
    std::cout << value_at(flat_coord) << '\n';
    return 0;
}
