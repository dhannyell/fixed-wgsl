// Returns floor((a_mag * 65536) / d) as (lo, hi).
// Q16 magnitudes: a_mag <= 2^31 and 1 <= d <= 2^31.
fn udiv48_32(a_mag: u32, d:u32) -> vec2<u32> {
    let whole = a_mag / d;
    let rem = a_mag - whole * d;

    // Normalize for a base 65536 quotient estimate.
    let shift = countLeadingZeros(d);
    let norm_d = d << shift;
    let hi = norm_d >> 16u;
    let lo = norm_d & 0xFFFFu;
    let norm_rem = rem << shift;

    var frac = norm_rem / hi;
    var estimate_rem = norm_rem - frac * hi;

    if (frac >= 65536u ||
        frac * lo > (estimate_rem << 16u)) {
            frac -= 1u;
            estimate_rem += hi;

            if (estimate_rem < 65536u && 
            (frac >= 65536u || frac * lo > (estimate_rem << 16u))) {
                frac -= 1u;
            }
    }

    return vec2<u32>(
        (whole << 16u) | frac,
        whole >> 16u,
    );
}

fn umul_32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;

    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);

    let lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);

    return vec2<u32>(lo, hi);
}

fn mag32(v: i32) -> u32 {
    let u = bitcast<u32>(v);
    return select(u, 0u - u, v < 0);
}

fn q16_square_le_scaled_raw(root: u32, raw: u32) -> bool {
    let square = umul_32(root, root);
    let lo = raw << 16u;
    let hi = raw >> 16u;

    return square.y < hi ||
        (square.y == hi && square.x <= lo);
}

fn q16_mul(a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q16 {
    let neg = (a.v < 0) != (b.v < 0);
    let p = umul_32(mag32(a.v), mag32(b.v));

    let trunc_lo = (p.x >> 16u) | (p.y << 16u);
    let trunc_hi = p.y >> 16u;

    let round_down = neg && ((p.x & 0xFFFFu) != 0u);
    let mag_lo = trunc_lo + u32(round_down);
    let mag_hi = trunc_hi + u32(round_down && (mag_lo == 0u));

    let limit = select(0x7fffffffu, 0x80000000u, neg);
    let over = (mag_hi != 0u) || (mag_lo > limit);

    (*sat).count += u32(over);

    let finite_raw = select(
        bitcast<i32>(mag_lo),
        bitcast<i32>(0u - mag_lo),
        neg,
    );
    let sat_raw = select(Q16_MAX.v, Q16_MIN.v, neg);

    return Q16(select(finite_raw, sat_raw, over));
}

fn q16_div(a: Q16, b: Q16, sat: ptr<function, Sat>) -> Q16 {
    let div_zero = b.v == 0;
    (*sat).fault += u32(div_zero);

    let neg = (a.v < 0) != (b.v < 0);
    let d   = select(mag32(b.v), 1u, div_zero);
    let q   = udiv48_32(mag32(a.v), d);

    // Negative results may reach 2^31 in magnitude; positive stop at 2^31-1.
    let limit = select(bitcast<u32>(Q16_MAX.v), 2147483648u, neg);
    let over  = (q.y != 0u) || (q.x > limit) || div_zero;
    (*sat).count += u32(over && !div_zero);

    let mag = select(q.x, limit, over);
    let r   = select(i32(mag), -i32(mag), neg); // i32(2^31) wraps to MIN; -MIN is MIN
    return Q16(r);
}