// Returns floor((a_mag * 65536) / d) as (lo, hi).
// Q16 magnitudes: a_mag <= 2^31 and 1 <= d <= 2^31.
fn udiv48_32(a_mag: u32, d: u32) -> vec2<u32> {
    let whole = a_mag / d;
    let rem = a_mag - whole * d;
    let shift = countLeadingZeros(d);
    let high = (d << shift) >> 16u;
    var fraction = (rem << shift) / high;
    // The normalized estimate is at most two above the exact fractional digit.
    let numerator = u64(rem) << 16u;
    var product = u64(fraction) * u64(d);
    if (product > numerator) {
        fraction -= 1u;
        product -= u64(d);
        if (product > numerator) {
            fraction -= 1u;
        }
    }
    return vec2<u32>((whole << 16u) | fraction, whole >> 16u);
}

fn mag32(v: i32) -> u32 {
    let u = bitcast<u32>(v);
    return select(u, 0u - u, v < 0);
}

fn q16_mul(a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q16 {
    let p = (i64(a.v) * i64(b.v)) >> 16;
    let hi = p > i64(Q16_MAX.v);
    let lo = p < i64(Q16_MIN.v);

    (*sat).count += u32(hi || lo);
    return Q16(i32(select(select(p, i64(Q16_MAX.v), hi), i64(Q16_MIN.v), lo)));
}

// q16_mul_round rounds the product to the nearest step, ties toward positive
// infinity. The arithmetic shift makes the +2^15 bias exact for both signs.
fn q16_mul_round(a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q16 {
    let p = (i64(a.v) * i64(b.v) + 32768li) >> 16;
    let hi = p > i64(Q16_MAX.v);
    let lo = p < i64(Q16_MIN.v);

    (*sat).count += u32(hi || lo);
    return Q16(i32(select(select(p, i64(Q16_MAX.v), hi), i64(Q16_MIN.v), lo)));
}

fn q16_div(a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q16 {
    let div_zero = b.v == 0;
    (*sat).fault += u32(div_zero);

    let neg = (a.v < 0) != (b.v < 0);
    let d = select(mag32(b.v), 1u, div_zero);
    let q = udiv48_32(mag32(a.v), d);

    // Negative results may reach 2^31 in magnitude; positive stop at 2^31-1.
    let limit = select(bitcast<u32>(Q16_MAX.v), 2147483648u, neg);
    let over = (q.y != 0u) || (q.x > limit) || div_zero;
    (*sat).count += u32(over && !div_zero);

    let mag = select(q.x, limit, over);
    let r = select(i32(mag), -i32(mag), neg); // i32(2^31) wraps to MIN; -MIN is MIN
    return Q16(r);
}

fn q16_square_le_scaled_raw(root: u32, raw: u32) -> bool {
    return u64(root) * u64(root) <= (u64(raw) << 16u);
}

// Use u64 here because unsigned overflow wraps consistently in HLSL.
// Signed overflow is undefined, so DXC may remove overflow checks.
// Read the sign from the i32 high word instead of shifting a 64-bit value.
fn q48_to_u64(a: Q48) -> u64 {
    return (u64(bitcast<u32>(a.hi)) << 32u) | u64(a.lo);
}

fn q48_from_u64(x: u64) -> Q48 {
    return Q48(u32(x & 4294967295lu), bitcast<i32>(u32(x >> 32u)));
}

fn q48_from_i64(x: i64) -> Q48 {
    return Q48(u32(x & 4294967295li), i32(x >> 32u));
}

fn q48_add(a: Q48, b: Q48, sat: ptr<function, Sat>) -> Q48 {
    let res = q48_from_u64(q48_to_u64(a) + q48_to_u64(b));
    let neg = a.hi < 0;
    let overflow = (neg == (b.hi < 0)) && ((res.hi < 0) != neg);
    (*sat).count += u32(overflow);
    return q48_saturate(res, neg, overflow);
}

fn q48_sub(a: Q48, b: Q48, sat: ptr<function, Sat>) -> Q48 {
    let res = q48_from_u64(q48_to_u64(a) - q48_to_u64(b));
    let neg = a.hi < 0;
    let overflow = (neg != (b.hi < 0)) && ((res.hi < 0) != neg);
    (*sat).count += u32(overflow);
    return q48_saturate(res, neg, overflow);
}

fn q48_mul_add16(q: Q48, a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q48 {
    // The product has at most 62 bits; only the add can saturate.
    let p = (i64(a.v) * i64(b.v)) >> 16u;
    return q48_add(q, q48_from_i64(p), sat);
}