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
    let p = i64(a.v) * i64(b.v) >> 16;
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

fn q48_add(a: Q48, b: Q48, sat: ptr<function, Sat>) -> Q48 {
    // u32 -> i64 sign-extends on some backends; go through u64 to zero-extend.
    let a_raw = (i64(a.hi) << 32u) | i64(u64(a.lo));
    let b_raw = (i64(b.hi) << 32u) | i64(u64(b.lo));
    
    let res = a_raw + b_raw;
    
    let overflow = ((a_raw >= 0) == (b_raw >= 0)) && ((res >= 0) != (a_raw >= 0));
    (*sat).count += u32(overflow);
    
    let neg = a_raw < 0;
    let sat_lo = select(Q48_MAX.lo, Q48_MIN.lo, neg);
    let sat_hi = select(Q48_MAX.hi, Q48_MIN.hi, neg);
    
    let res_lo = u32(res & 4294967295);
    
    let res_hi = i32(res >> 32u);
    
    return Q48(
        select(res_lo, sat_lo, overflow), 
        select(res_hi, sat_hi, overflow)
    );
}