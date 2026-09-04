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

    var root = u32(sqrt(f32(raw)) * 256.0);

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