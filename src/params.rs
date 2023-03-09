use num_traits::{AsPrimitive, Float};

use crate::stl::{stl, Bound};
use crate::Error;

#[derive(Debug)]
pub struct StlParams {
    ns: Option<usize>,
    nt: Option<usize>,
    nl: Option<usize>,
    isdeg: i32,
    itdeg: i32,
    ildeg: Option<i32>,
    nsjump: Option<usize>,
    ntjump: Option<usize>,
    nljump: Option<usize>,
    fastjump: Option<bool>,
    ni: Option<usize>,
    no: Option<usize>,
    robust: bool,
}

#[derive(Debug)]
pub struct StlResult<T: Float + 'static> {
    pub seasonal: Vec<T>,
    pub trend: Vec<T>,
    pub remainder: Vec<T>,
    pub weights: Vec<T>,
}

pub fn params() -> StlParams {
    StlParams {
        ns: None,
        nt: None,
        nl: None,
        isdeg: 0,
        itdeg: 1,
        ildeg: None,
        nsjump: None,
        ntjump: None,
        nljump: None,
        fastjump: None,
        ni: None,
        no: None,
        robust: false,
    }
}

impl StlParams {
    pub fn seasonal_length(&mut self, ns: usize) -> &mut Self {
        self.ns = Some(ns);
        self
    }

    pub fn trend_length(&mut self, nt: usize) -> &mut Self {
        self.nt = Some(nt);
        self
    }

    pub fn low_pass_length(&mut self, nl: usize) -> &mut Self {
        self.nl = Some(nl);
        self
    }

    pub fn seasonal_degree(&mut self, isdeg: i32) -> &mut Self {
        self.isdeg = isdeg;
        self
    }

    pub fn trend_degree(&mut self, itdeg: i32) -> &mut Self {
        self.itdeg = itdeg;
        self
    }

    pub fn low_pass_degree(&mut self, ildeg: i32) -> &mut Self {
        self.ildeg = Some(ildeg);
        self
    }

    pub fn seasonal_jump(&mut self, nsjump: usize) -> &mut Self {
        self.nsjump = Some(nsjump);
        self
    }

    pub fn trend_jump(&mut self, ntjump: usize) -> &mut Self {
        self.ntjump = Some(ntjump);
        self
    }

    pub fn low_pass_jump(&mut self, nljump: usize) -> &mut Self {
        self.nljump = Some(nljump);
        self
    }

    pub fn inner_loops(&mut self, ni: usize) -> &mut Self {
        self.ni = Some(ni);
        self
    }

    pub fn outer_loops(&mut self, no: usize) -> &mut Self {
        self.no = Some(no);
        self
    }

    pub fn fast_jump(&mut self, fastjump: bool) -> &mut Self {
        self.fastjump = Some(fastjump);
        self
    }

    pub fn robust(&mut self, robust: bool) -> &mut Self {
        self.robust = robust;
        self
    }

    pub fn fit<T: Bound + 'static>(&self, y: &[T], np: usize) -> Result<StlResult<T>, Error>
    where
        usize: AsPrimitive<T>,
    {
        let n = y.len();

        if n < np * 2 {
            return Err(Error::Series(
                "series has less than two periods".to_string(),
            ));
        }

        let ns = self.ns.unwrap_or(np);

        let isdeg = self.isdeg;
        let itdeg = self.itdeg;

        let mut rw = vec![T::zero(); n];
        let mut season = vec![T::zero(); n];
        let mut trend = vec![T::zero(); n];

        let ildeg = self.ildeg.unwrap_or(itdeg);
        let mut newns = ns.max(3);
        if newns % 2 == 0 {
            newns += 1;
        }

        let newnp = np.max(2);
        let mut nt = ((1.5 * newnp as f32) / (1.0 - 1.5 / newns as f32)).ceil() as usize;
        nt = self.nt.unwrap_or(nt);
        nt = nt.max(3);
        if nt % 2 == 0 {
            nt += 1;
        }

        let mut nl = self.nl.unwrap_or(newnp);
        if nl % 2 == 0 && self.nl.is_none() {
            nl += 1;
        }

        let ni = self.ni.unwrap_or(if self.robust { 1 } else { 2 });
        let no = self.no.unwrap_or(if self.robust { 15 } else { 0 });

        let jump_factor = if self.fastjump.unwrap_or(false) {
            5.0
        } else {
            10.0
        };
        let nsjump = self
            .nsjump
            .unwrap_or(((newns as f32) / jump_factor).ceil() as usize);
        let ntjump = self
            .ntjump
            .unwrap_or(((nt as f32) / jump_factor).ceil() as usize);
        let nljump = self
            .nljump
            .unwrap_or(((nl as f32) / jump_factor).ceil() as usize);

        if newns < 3 {
            return Err(Error::Parameter(
                "seasonal_length must be at least 3".to_string(),
            ));
        }
        if nt < 3 {
            return Err(Error::Parameter(
                "trend_length must be at least 3".to_string(),
            ));
        }
        if nl < 3 {
            return Err(Error::Parameter(
                "low_pass_length must be at least 3".to_string(),
            ));
        }
        if newnp < 2 {
            return Err(Error::Parameter("period must be at least 2".to_string()));
        }

        if isdeg != 0 && isdeg != 1 {
            return Err(Error::Parameter(
                "seasonal_degree must be 0 or 1".to_string(),
            ));
        }
        if itdeg != 0 && itdeg != 1 {
            return Err(Error::Parameter("trend_degree must be 0 or 1".to_string()));
        }
        if ildeg != 0 && ildeg != 1 {
            return Err(Error::Parameter(
                "low_pass_degree must be 0 or 1".to_string(),
            ));
        }

        if newns % 2 != 1 {
            return Err(Error::Parameter("seasonal_length must be odd".to_string()));
        }
        if nt % 2 != 1 {
            return Err(Error::Parameter("trend_length must be odd".to_string()));
        }
        if nl % 2 != 1 {
            return Err(Error::Parameter("low_pass_length must be odd".to_string()));
        }

        stl(
            y,
            n,
            newnp,
            newns,
            nt,
            nl,
            isdeg,
            itdeg,
            ildeg,
            nsjump,
            ntjump,
            nljump,
            ni,
            no,
            &mut rw,
            &mut season,
            &mut trend,
        );

        let mut remainder = Vec::with_capacity(n);
        for i in 0..n {
            remainder.push(y[i] - season[i] - trend[i]);
        }

        Ok(StlResult {
            seasonal: season,
            trend,
            remainder,
            weights: rw,
        })
    }
}

fn var<T: Bound + std::iter::Sum<T> + 'static>(series: &[T]) -> T
where
    usize: AsPrimitive<T>,
{
    let mean = series.iter().copied().sum::<T>() / series.len().as_();
    series
        .iter()
        .map(|&v| (v - mean).powf(<T as From<f32>>::from(2.0)))
        .sum::<T>()
        / (series.len().as_() - T::one())
}

impl<T: Bound + std::iter::Sum<T> + 'static> StlResult<T>
where
    usize: AsPrimitive<T>,
{
    pub fn seasonal(&self) -> &Vec<T> {
        &self.seasonal
    }

    pub fn trend(&self) -> &Vec<T> {
        &self.trend
    }

    pub fn remainder(&self) -> &Vec<T> {
        &self.remainder
    }

    pub fn weights(&self) -> &Vec<T> {
        &self.weights
    }

    pub fn seasonal_strength(&self) -> T {
        let sr = self
            .seasonal()
            .iter()
            .zip(self.remainder())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<T>>();
        (T::one() - var(self.remainder()) / var(&sr)).max(T::zero())
    }

    pub fn trend_strength(&self) -> T {
        let tr = self
            .trend()
            .iter()
            .zip(self.remainder())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<T>>();
        (T::one() - var(self.remainder()) / var(&tr)).max(T::zero())
    }
}
