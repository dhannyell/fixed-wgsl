// Q16.16 in an i32. Overflow saturates. Same bits as fixed.Q16.

struct Q16 { v: i32 }

const Q16_ONE = Q16(65536);
const Q16_MIN = Q16(-2147483648);
const Q16_MAX = Q16(2147483647);

// Development diagnostics. The host sets it to true when it creates the
// pipeline. Values never depend on it; only the counter write-out does.
override COUNT_SATURATION: bool = false;

// One saturation record per invocation. The host reduces it to one counter
// update per dispatch, so the count matches a scalar loop.
struct Sat { count: u32, fault: u32}

fn q16_add(a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q16 {
    let s = a.v + b.v;

    let pos_overflow = (a.v > 0) && (b.v > 0) && (s < 0);
    let neg_overflow = (a.v < 0) && (b.v < 0) && (s >= 0);

    (*sat).count += u32(pos_overflow || neg_overflow);

    let res = select(s, Q16_MAX.v, pos_overflow);
    return Q16(select(res, Q16_MIN.v, neg_overflow));
}

fn q16_sub(a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q16 {
    let s = a.v - b.v;

    let pos_overflow = (a.v >= 0) && (b.v < 0) && (s < 0);
    let neg_overflow = (a.v < 0) && (b.v > 0) && (s >= 0);

    (*sat).count += u32(pos_overflow || neg_overflow);

    let res = select(s, Q16_MAX.v, pos_overflow);
    return Q16(select(res, Q16_MIN.v, neg_overflow));
}

fn q16_sqrt(a: Q16, sat: ptr<function, Sat>) -> Q16 {
    if (a.v < 0) {
        (*sat).fault += 1u;
        return Q16(0);
    }

    let raw = u32(a.v);

    // The f32 estimate may be a little above or below the actual root.
    // Adjust it down or up until we find the largest root that fits.
    var root = u32(sqrt(f32(raw)) * 256.0);

    while (root > 0u && !q16_square_le_scaled_raw(root, raw)) {
        root -= 1u;
    }

    while (q16_square_le_scaled_raw(root + 1u, raw)) {
        root += 1u;
    }

    return Q16(i32(root));
}

fn q16_neg(a: Q16, sat: ptr<function, Sat>) -> Q16 {
    let is_min = a.v == Q16_MIN.v;
    (*sat).count += u32(is_min);
    return Q16(select(-a.v, Q16_MAX.v, is_min));
}

fn q16_clamp(a: Q16, lo: Q16, hi: Q16) -> Q16 {
    return Q16(clamp(a.v, lo.v, hi.v));
}

fn q16_floor(a: Q16) -> Q16 {
    return Q16(a.v & ~(Q16_ONE.v - 1));
}

fn q16_min(a: Q16, b:Q16) -> Q16 {
    return Q16(min(a.v, b.v));
}

fn q16_max(a:Q16, b:Q16) -> Q16 {
    return Q16(max(a.v, b.v));
}

fn q16_greater(a:Q16, b:Q16) -> bool {
    return a.v > b.v;
}

fn q16_eq(a:Q16, b:Q16) -> bool {
    return a.v == b.v;
}

fn q16_blend(a:Q16, b:Q16, m:bool) -> Q16 {
    return Q16(select(b.v, a.v, m));
}

fn q16_saturate(a: Q16, neg:bool, overflow:bool) -> Q16 {
    let res = select(a.v, Q16_MAX.v, overflow);
    return Q16(select(res, Q16_MIN.v, neg));
}

fn q16_to_q48(a: Q16) -> Q48 {
    // Sign extension never saturates: hi is all ones for negative values.
    return Q48(bitcast<u32>(a.v), a.v >> 31u);
}

struct Q32 {
    lo: u32,
    hi: i32,
}

const Q32_ZERO = Q32(0u, 0);
const Q32_ONE  = Q32(0u, 1);
const Q32_HALF = Q32(0x80000000u, 0);
const Q32_MIN  = Q32(0u, -0x80000000);
const Q32_MAX  = Q32(0xffffffffu, 0x7fffffff);

fn q32_add(a: Q32, b: Q32, sat: ptr<function, Sat>) -> Q32 {
    let lo = a.lo + b.lo;
    let carry = u32(lo < a.lo);
    let hi = a.hi + b.hi + i32(carry);

    let pos_overflow = (a.hi >= 0) && (b.hi >= 0) && (hi < 0);
    let neg_overflow = (a.hi < 0) && (b.hi < 0) && (hi >= 0);
    
    (*sat).count += u32(pos_overflow || neg_overflow);

    return Q32(
        select(select(lo, Q32_MAX.lo, pos_overflow), Q32_MIN.lo, neg_overflow),
        select(select(hi, Q32_MAX.hi, pos_overflow), Q32_MIN.hi, neg_overflow),
    );
}

fn q32_saturate(v:Q32, neg:bool, overflow:bool) -> Q32 {
    let lo = select(Q32_MAX.lo, Q32_MIN.lo,neg);
    let hi = select(Q32_MAX.hi, Q32_MIN.hi, neg);
    return Q32(select(v.lo, lo, overflow), select(v.hi,hi, overflow));
}

fn q32_to_q16(a:Q32, sat: ptr<function, Sat>) -> Q16 {
    return q48_to_q16(q32_to_q48(a), sat);
}

fn q32_to_q48(a:Q32) -> Q48 {
    let lo = (a.lo >> 16u) | (bitcast<u32>(a.hi) << 16u);
    let hi = a.hi >> 16u;
    return Q48(lo,hi);
}

struct Q48 {
    lo: u32,
    hi: i32,
}

const Q48_ZERO = Q48(0u, 0);
const Q48_ONE  = Q48(0x10000u, 0);
const Q48_HALF = Q48(0x8000u, 0);
const Q48_MIN  = Q48(0u, -0x80000000);
const Q48_MAX  = Q48(0xffffffffu, 0x7fffffff);

// MAX when the first operand is non-negative, MIN otherwise; same as fixed.
fn q48_saturate(v: Q48, neg: bool, overflow: bool) -> Q48 {
    let lo = select(Q48_MAX.lo, Q48_MIN.lo, neg);
    let hi = select(Q48_MAX.hi, Q48_MIN.hi, neg);
    return Q48(select(v.lo, lo, overflow), select(v.hi, hi, overflow));
}

fn q48_to_q16(a: Q48, sat: ptr<function, Sat>) -> Q16 {
    let lo = bitcast<i32>(a.lo);
    let fits = a.hi == (lo >> 31);
    (*sat).count += u32(!fits);
    let limit = select(Q16_MAX.v, Q16_MIN.v, a.hi < 0);
    return Q16(select(limit, lo, fits));
}

fn q48_to_q32(a:Q48, sat: ptr<function, Sat>) -> Q32 {
    let top = a.hi >> 15u;
    let overflow = (top != 0) && (top != -1);
    (*sat).count += u32(overflow);
    let lo = a.lo << 16u;
    let hi = (a.hi << 16u) | i32(a.lo >> 16u);
    return q32_saturate(Q32(lo,hi), a.hi < 0, overflow);
}