// Ported from https://www.netlib.org/a/stl
//
// Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
// STL: A Seasonal-Trend Decomposition Procedure Based on Loess.
// Journal of Official Statistics, 6(1), 3-33.

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, Mul, MulAssign},
};

use num_traits::{AsPrimitive, Float};
use tracing::instrument;

pub trait Bound: Float + MulAssign + AddAssign + DivAssign + Mul + From<f32> + Sum {}
impl<T> Bound for T where T: Float + MulAssign + AddAssign + DivAssign + Mul + From<f32> + Sum {}

#[instrument(skip(
    y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, no, rw, season, trend
))]
pub fn stl<T: Bound + 'static>(
    // Input time series.
    y: &[T],
    // Length of y.
    n: usize,
    // Number of observations per seasonal period.
    np: usize,
    // Smoothing parameter for the seasonal filter.
    ns: usize,
    // Smoothing parameter for the trend filter.
    nt: usize,
    // Smoothing parameter for the low-pass filter.
    nl: usize,
    // Seasonal filter degree.
    isdeg: i32,
    // Trend filter degree.
    itdeg: i32,
    // Low-pass filter degree.
    ildeg: i32,
    nsjump: usize,
    ntjump: usize,
    nljump: usize,
    // Number of inner loop iterations.
    ni: usize,
    // Number of outer loop (robustness) iterations.
    no: usize,
    // Robustness weights.
    rw: &mut [T],
    season: &mut [T],
    trend: &mut [T],
) where
    usize: AsPrimitive<T>,
{
    let mut work1 = vec![T::zero(); n + 2 * np];
    let mut work2 = vec![T::zero(); n + 2 * np];
    let mut work3 = vec![T::zero(); n + 2 * np];
    let mut work4 = vec![T::zero(); n + 2 * np];
    let mut work5 = vec![T::zero(); n + 2 * np];

    let mut userw = false;
    let mut k = 0;

    loop {
        onestp(
            y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, userw, rw,
            season, trend, &mut work1, &mut work2, &mut work3, &mut work4, &mut work5,
        );
        k += 1;
        if k > no {
            break;
        }
        for i in 0..n {
            work1[i] = trend[i] + season[i];
        }
        rwts(y, n, &work1, rw);
        userw = true;
    }

    if no == 0 {
        rw.iter_mut().take(n).for_each(|x| *x = T::one());
    }
}

#[instrument(skip(y, n, len, ideg, njump, userw, rw, ys, res))]
fn ess<T: Bound + 'static>(
    y: &[T],
    n: usize,
    len: usize,
    ideg: i32,
    njump: usize,
    userw: bool,
    rw: &[T],
    ys: &mut [T],
    res: &mut [T],
) where
    usize: AsPrimitive<T>,
{
    if n < 2 {
        ys[0] = y[0];
        return;
    }

    let mut nleft = 0;
    let mut nright = 0;

    let newnj = njump.min(n - 1);
    if len >= n {
        let _span = tracing::info_span!("ess len >= n").entered();
        nleft = 1;
        nright = n;
        let mut i = 1;
        while i <= n {
            let ok = est(
                y,
                n,
                len,
                ideg,
                i.as_(),
                &mut ys[i - 1],
                nleft,
                nright,
                res,
                userw,
                rw,
            );
            if !ok {
                ys[i - 1] = y[i - 1];
            }
            i += newnj;
        }
    } else if newnj == 1 {
        let _span = tracing::info_span!("ess newnj == 1").entered();
        // newnj equal to one, len less than n
        let nsh = (len + 1) / 2;
        nleft = 1;
        nright = len;
        for i in 1..=n {
            // fitted value at i
            if i > nsh && nright != n {
                nleft += 1;
                nright += 1;
            }
            let ok = est(
                y,
                n,
                len,
                ideg,
                i.as_(),
                &mut ys[i - 1],
                nleft,
                nright,
                res,
                userw,
                rw,
            );
            if !ok {
                ys[i - 1] = y[i - 1];
            }
        }
    } else {
        // newnj greater than one, len less than n
        let _span = tracing::info_span!("ess newnj > 1, len < n").entered();
        let nsh = (len + 1) / 2;
        let mut i = 1;
        while i <= n {
            // fitted value at i
            if i < nsh {
                nleft = 1;
                nright = len;
            } else if i > n - nsh {
                nleft = n - len + 1;
                nright = n;
            } else {
                nleft = i - nsh + 1;
                nright = len + i - nsh;
            }
            let ok = est(
                y,
                n,
                len,
                ideg,
                i.as_(),
                &mut ys[i - 1],
                nleft,
                nright,
                res,
                userw,
                rw,
            );
            if !ok {
                ys[i - 1] = y[i - 1];
            }
            i += newnj;
        }
    }

    if newnj != 1 {
        let _span1 = tracing::info_span!("ess newnj != 1").entered();
        let mut i = 1;
        let span2 = tracing::info_span!("ess newnj != 1: while loop").entered();
        while i <= n - newnj {
            let delta = (ys[i + newnj - 1] - ys[i - 1]) / newnj.as_();
            for j in i + 1..=i + newnj - 1 {
                ys[j - 1] = ys[i - 1] + delta * (j - i).as_();
            }
            i += newnj;
        }
        drop(span2);
        let k = ((n - 1) / newnj) * newnj + 1;
        if k != n {
            let _span = tracing::info_span!("ess newnj != 1: k != n").entered();
            let ok = est(
                y,
                n,
                len,
                ideg,
                n.as_(),
                &mut ys[n - 1],
                nleft,
                nright,
                res,
                userw,
                rw,
            );
            if !ok {
                ys[n - 1] = y[n - 1];
                if k != n - 1 {
                    let delta = (ys[n - 1] - ys[k - 1]) / (n - k).as_();
                    for j in k + 1..=n - 1 {
                        ys[j - 1] = ys[k - 1] + delta * (j - k).as_();
                    }
                }
            }
        }
    }
}

#[instrument(skip(y, n, len, ideg, xs, ys, nleft, nright, w, userw, rw))]
fn est<T: Bound + 'static>(
    y: &[T],
    n: usize,
    len: usize,
    ideg: i32,
    xs: T,
    ys: &mut T,
    nleft: usize,
    nright: usize,
    w: &mut [T],
    userw: bool,
    rw: &[T],
) -> bool
where
    usize: AsPrimitive<T>,
{
    let range = n.as_() - T::one();
    let mut h = (xs - nleft.as_()).max(nright.as_() - xs);

    if len > n {
        h += ((len - n) / 2).as_();
    }

    let h9 = <T as From<f32>>::from(0.999) * h;
    let h1 = <T as From<f32>>::from(0.001) * h;

    // compute weights
    let mut a = T::zero();
    for j in nleft..=nright {
        w[j - 1] = T::zero();
        let r = (j.as_() - xs).abs();
        if r <= h9 {
            if r <= h1 {
                w[j - 1] = T::one();
            } else {
                w[j - 1] = (T::one() - (r / h).powi(3)).powi(3);
            }
            if userw {
                w[j - 1] *= rw[j - 1];
            }
            a += w[j - 1];
        }
    }

    if a <= T::zero() {
        false
    } else { // weighted least squares
        for j in nleft..=nright { // make sum of w(j) == 1
            w[j - 1] /= a;
        }

        if h > T::zero() && ideg > 0 { // use linear fit
            let mut a = T::zero();
            for j in nleft..=nright { // weighted center of x values
                a += w[j - 1] * j.as_();
            }
            let mut b = xs - a;
            let mut c = T::zero();
            for j in nleft..=nright {
                c += w[j - 1] * (j.as_() - a).powi(2);
            }
            if c.sqrt() > <T as From<f32>>::from(0.001) * range {
                b /= c;

                // points are spread out enough to compute slope
                for j in nleft..=nright {
                    w[j - 1] *= b * (j.as_() - a) + 1.0.into();
                }
            }
        }

        *ys = 0.0.into();
        for j in nleft..=nright {
            *ys += w[j - 1] * y[j - 1];
        }

        true
    }
}

#[instrument(skip(x, n, np, trend, work))]
fn fts<T: Bound + 'static>(x: &[T], n: usize, np: usize, trend: &mut [T], work: &mut [T])
where
    usize: AsPrimitive<T>,
{
    ma(x, n, np, trend);
    ma(trend, n - np + 1, np, work);
    ma(work, n - 2 * np + 2, 3, trend);
}

#[instrument(skip(x, n, len, ave))]
fn ma<T: Bound + 'static>(x: &[T], n: usize, len: usize, ave: &mut [T])
where
    usize: AsPrimitive<T>,
{
    let newn = n - len + 1;
    let flen = len.as_();
    let mut v = T::zero();

    // get the first average
    for x_i in &x[..len] {
        v += *x_i;
    }

    ave[0] = v / flen;
    if newn > 1 {
        let mut k = len;
        for (m, a) in &mut ave[1..newn].iter_mut().enumerate() {
            // window down the array
            v = v - x[m] + x[k];
            *a = v / flen;
            k += 1;
        }
    }
}

#[instrument(skip(
    y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, userw, rw, season,
    trend, work1, work2, work3, work4, work5
))]
fn onestp<T: Bound + 'static>(
    y: &[T],
    n: usize,
    np: usize,
    ns: usize,
    nt: usize,
    nl: usize,
    isdeg: i32,
    itdeg: i32,
    ildeg: i32,
    nsjump: usize,
    ntjump: usize,
    nljump: usize,
    ni: usize,
    userw: bool,
    rw: &mut [T],
    season: &mut [T],
    trend: &mut [T],
    work1: &mut [T],
    work2: &mut [T],
    work3: &mut [T],
    work4: &mut [T],
    work5: &mut [T],
) where
    usize: AsPrimitive<T>,
{
    for _ in 0..ni {
        // Detrend.
        for i in 0..n {
            work1[i] = y[i] - trend[i];
        }

        ss(
            work1, n, np, ns, isdeg, nsjump, userw, rw, work2, work3, work4, work5, season,
        );
        fts(work2, n + 2 * np, np, work3, work1);
        ess(work3, n, nl, ildeg, nljump, false, work4, work1, work5);
        for i in 0..n {
            season[i] = work2[np + i] - work1[i];
        }
        // Deseasonalise.
        for i in 0..n {
            work1[i] = y[i] - season[i];
        }
        // Trend smoothing.
        ess(work1, n, nt, itdeg, ntjump, userw, rw, trend, work3);
    }
}

#[instrument(skip(y, fit, rw))]
fn rwts<T: Bound>(y: &[T], n: usize, fit: &[T], rw: &mut [T]) {
    for i in 0..n {
        rw[i] = (y[i] - fit[i]).abs();
    }

    let mid1 = (n - 1) / 2;
    let mid2 = n / 2;

    rw.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let cmad = <T as From<f32>>::from(3.0) * (rw[mid1] + rw[mid2]); // 6 * median abs resid
    let c9 = <T as From<f32>>::from(0.999) * cmad;
    let c1 = <T as From<f32>>::from(0.001) * cmad;

    for i in 0..n {
        let r = (y[i] - fit[i]).abs();
        if r <= c1 {
            rw[i] = T::one();
        } else if r <= c9 {
            rw[i] = (T::one() - (r / cmad).powi(2)).powi(2);
        } else {
            rw[i] = T::zero();
        }
    }
}

/// Seasonal smoothing.
#[instrument(skip(
    y, n, np, ns, isdeg, nsjump, userw, rw, season, work1, work2, work3, work4
))]
fn ss<T: Bound + 'static>(
    y: &[T],
    n: usize,
    np: usize,
    ns: usize,
    isdeg: i32,
    nsjump: usize,
    userw: bool,
    rw: &[T],
    season: &mut [T],
    work1: &mut [T],
    work2: &mut [T],
    work3: &mut [T],
    work4: &mut [T],
) where
    usize: AsPrimitive<T>,
{
    for j in 1..=np {
        let k = (n - j) / np + 1;

        for i in 1..=k {
            work1[i - 1] = y[(i - 1) * np + j - 1];
        }
        if userw {
            for i in 1..=k {
                work3[i - 1] = rw[(i - 1) * np + j - 1];
            }
        }
        ess(
            work1,
            k,
            ns,
            isdeg,
            nsjump,
            userw,
            work3,
            &mut work2[1..],
            work4,
        );
        let mut xs = T::zero();
        let nright = ns.min(k);
        let ok = est(
            work1,
            k,
            ns,
            isdeg,
            xs,
            &mut work2[0],
            1,
            nright,
            work4,
            userw,
            work3,
        );
        if !ok {
            work2[0] = work2[1];
        }
        xs = (k + 1).as_();
        let nleft = 1.max(k as i32 - ns as i32 + 1) as usize;
        let ok = est(
            work1,
            k,
            ns,
            isdeg,
            xs,
            &mut work2[k + 1],
            nleft,
            k,
            work4,
            userw,
            work3,
        );
        if !ok {
            work2[k + 1] = work2[k];
        }
        for m in 1..=k + 2 {
            season[(m - 1) * np + j - 1] = work2[m - 1];
        }
    }
}
