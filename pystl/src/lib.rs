use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyException, prelude::*};
use stlrs::{params, StlParams};

#[derive(Debug)]
#[pyclass]
pub struct PySTL {
    stl: StlParams,
    y: Vec<f64>,
    np: usize,
}

#[pymethods]
impl PySTL {
    #[new]
    pub fn new(
        y: PyReadonlyArrayDyn<'_, f64>,
        np: usize,
        seasonal_length: Option<usize>,
        trend_length: Option<usize>,
        low_pass_length: Option<usize>,
        seasonal_degree: Option<i32>,
        trend_degree: Option<i32>,
        low_pass_degree: Option<i32>,
        seasonal_jump: Option<usize>,
        trend_jump: Option<usize>,
        low_pass_jump: Option<usize>,
        outer_loops: Option<usize>,
        inner_loops: Option<usize>,
        fast_jump: Option<bool>,
        robust: Option<bool>,
    ) -> PyResult<Self> {
        let mut stl = params();
        if let Some(x) = seasonal_length {
            stl.seasonal_length(x);
        }
        if let Some(x) = trend_length {
            stl.trend_length(x);
        }
        if let Some(x) = low_pass_length {
            stl.low_pass_length(x);
        }
        if let Some(x) = seasonal_degree {
            stl.seasonal_degree(x);
        }
        if let Some(x) = trend_degree {
            stl.trend_degree(x);
        }
        if let Some(x) = low_pass_degree {
            stl.low_pass_degree(x);
        }
        if let Some(x) = seasonal_jump {
            stl.seasonal_jump(x);
        }
        if let Some(x) = trend_jump {
            stl.trend_jump(x);
        }
        if let Some(x) = low_pass_jump {
            stl.low_pass_jump(x);
        }
        if let Some(x) = outer_loops {
            stl.outer_loops(x);
        }
        if let Some(x) = inner_loops {
            stl.inner_loops(x);
        }
        if let Some(x) = fast_jump {
            stl.fast_jump(x);
        }
        if let Some(x) = robust {
            stl.robust(x);
        }
        Ok(Self {
            stl,
            y: y.to_vec()?,
            np,
        })
    }

    pub fn seasonal_length(&mut self, ns: usize) {
        self.stl.seasonal_length(ns);
    }

    pub fn trend_length(&mut self, nt: usize) {
        self.stl.trend_length(nt);
    }

    pub fn low_pass_length(&mut self, nl: usize) {
        self.stl.low_pass_length(nl);
    }

    pub fn seasonal_degree(&mut self, isdeg: i32) {
        self.stl.seasonal_degree(isdeg);
    }

    pub fn trend_degree(&mut self, itdeg: i32) {
        self.stl.trend_degree(itdeg);
    }

    pub fn low_pass_degree(&mut self, ildeg: i32) {
        self.stl.low_pass_degree(ildeg);
    }

    pub fn seasonal_jump(&mut self, nsjump: usize) {
        self.stl.seasonal_jump(nsjump);
    }

    pub fn trend_jump(&mut self, ntjump: usize) {
        self.stl.trend_jump(ntjump);
    }

    pub fn low_pass_jump(&mut self, nljump: usize) {
        self.stl.low_pass_jump(nljump);
    }

    pub fn inner_loops(&mut self, ni: usize) {
        self.stl.inner_loops(ni);
    }

    pub fn outer_loops(&mut self, no: usize) {
        self.stl.outer_loops(no);
    }

    pub fn fast_jump(&mut self, fastjump: bool) {
        self.stl.fast_jump(fastjump);
    }

    pub fn robust(&mut self, robust: bool) {
        self.stl.robust(robust);
    }

    pub fn fit(&self) -> PyResult<StlFit> {
        self.stl
            .fit(&self.y, self.np)
            .map(StlFit::from_stl_result)
            .map_err(|x| PyException::new_err(x.to_string()))
    }
}

#[pyclass]
pub struct StlFit {
    seasonal: Py<PyArray1<f64>>,
    trend: Py<PyArray1<f64>>,
    remainder: Py<PyArray1<f64>>,
}

impl StlFit {
    fn from_stl_result(stl: stlrs::StlResult<f64>) -> Self {
        let (seasonal, trend, remainder, _) = stl.into_parts();
        Self {
            seasonal: Python::with_gil(|py| seasonal.into_pyarray(py).into()),
            trend: Python::with_gil(|py| trend.into_pyarray(py).into()),
            remainder: Python::with_gil(|py| remainder.into_pyarray(py).into()),
        }
    }
}

#[pymethods]
impl StlFit {
    pub fn seasonal(&self) -> &Py<PyArray1<f64>> {
        &self.seasonal
    }

    pub fn trend(&self) -> &Py<PyArray1<f64>> {
        &self.trend
    }

    pub fn remainder(&self) -> &Py<PyArray1<f64>> {
        &self.remainder
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn pystl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySTL>()?;
    m.add_class::<StlFit>()?;
    Ok(())
}
