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

/// Calculate the Seasonal-Trend decomposition of a time series.
#[pymethods]
impl PySTL {
    #[new]
    // Arguments here match statsmodels.tsa.seasonal.STL (plus a few extra)
    // so that the API matches, so ignore clippy's complaint.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        endog: PyReadonlyArrayDyn<'_, f64>,
        period: usize,
        seasonal: Option<usize>,
        trend: Option<usize>,
        low_pass: Option<usize>,
        seasonal_deg: Option<i32>,
        trend_deg: Option<i32>,
        low_pass_deg: Option<i32>,
        robust: Option<bool>,
        seasonal_jump: Option<usize>,
        trend_jump: Option<usize>,
        low_pass_jump: Option<usize>,
        fast_jump: Option<bool>,
    ) -> PyResult<Self> {
        let mut stl = params();
        if let Some(x) = seasonal {
            stl.seasonal_length(x);
        }
        if let Some(x) = trend {
            stl.trend_length(x);
        }
        if let Some(x) = low_pass {
            stl.low_pass_length(x);
        }
        if let Some(x) = seasonal_deg {
            stl.seasonal_degree(x);
        }
        if let Some(x) = trend_deg {
            stl.trend_degree(x);
        }
        if let Some(x) = low_pass_deg {
            stl.low_pass_degree(x);
        }
        if let Some(x) = robust {
            stl.robust(x);
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
        if let Some(x) = fast_jump {
            stl.fast_jump(x);
        }
        Ok(Self {
            stl,
            y: endog.to_vec()?,
            np: period,
        })
    }

    /// Fit the STL model to the data.
    pub fn fit(
        &mut self,
        outer_iter: Option<usize>,
        inner_iter: Option<usize>,
    ) -> PyResult<StlFit> {
        if let Some(x) = outer_iter {
            self.stl.outer_loops(x);
        }
        if let Some(x) = inner_iter {
            self.stl.inner_loops(x);
        }
        self.stl
            .fit(&self.y, self.np)
            .map(StlFit::from_stl_result)
            .map_err(|x| PyException::new_err(x.to_string()))
    }
}

/// A fitted STL model.
///
/// Contains the seasonal, trend, and remainder components of the decomposition.
/// These are all `numpy` arrays.
#[pyclass]
pub struct StlFit {
    /// The estimated seasonal component.
    #[pyo3(get)]
    seasonal: Py<PyArray1<f64>>,
    /// The estimated trend component.
    #[pyo3(get)]
    trend: Py<PyArray1<f64>>,
    /// The estimated residuals.
    #[pyo3(get)]
    resid: Py<PyArray1<f64>>,
}

impl StlFit {
    fn from_stl_result(stl: stlrs::StlResult<f64>) -> Self {
        Self {
            seasonal: Python::with_gil(|py| stl.seasonal.into_pyarray(py).into()),
            trend: Python::with_gil(|py| stl.trend.into_pyarray(py).into()),
            resid: Python::with_gil(|py| stl.remainder.into_pyarray(py).into()),
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn pystl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySTL>()?;
    m.add_class::<StlFit>()?;
    Ok(())
}
