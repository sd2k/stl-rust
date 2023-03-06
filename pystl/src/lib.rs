use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyException, prelude::*};
use stlrs::{params, StlParams};

#[derive(Debug, Clone)]
#[pyclass]
pub struct PySTL {
    stl: StlParams,
    y: Vec<f64>,
    np: usize,
}

#[pymethods]
impl PySTL {
    #[new]
    pub fn new(y: PyReadonlyArrayDyn<'_, f64>, np: usize) -> PyResult<Self> {
        Ok(Self {
            stl: params(),
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

