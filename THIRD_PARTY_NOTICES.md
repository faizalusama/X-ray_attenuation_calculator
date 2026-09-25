# Third-party notices

The X-ray Attenuation Workbench is distributed under the MIT License (see
[LICENSE](LICENSE)). That licence covers this project's own source only. The
components below keep their own terms, and redistributing the workbench
redistributes them too.

## Bundled in the source tree

| Component | Version | Licence | Location |
|---|---|---|---|
| Plotly.js | 4.0.0 | MIT — © 2016–2024 Plotly Technologies Inc. | [`static/vendor/plotly-4.0.0.min.js`](static/vendor/plotly-4.0.0.min.js), full text in [`static/vendor/PLOTLY-LICENSE.txt`](static/vendor/PLOTLY-LICENSE.txt) |

Plotly.js is vendored rather than loaded from a CDN so that the workbench runs
with no network access after installation. Replacing it with a different
version means replacing the accompanying licence text as well.

## Runtime dependencies

Installed from PyPI; not redistributed in this repository.

| Package | Licence |
|---|---|
| NumPy | BSD-3-Clause (with 0BSD, MIT, Zlib and CC0-1.0 components) |
| XrayDB | MIT |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| SciPy *(transitive via XrayDB; also a direct test dependency)* | BSD-3-Clause |
| SQLAlchemy *(transitive via XrayDB)* | MIT |

Exact verified versions are listed in [`requirements-lock.txt`](requirements-lock.txt).

## Scientific data

Cross-sections come from the **Elam, Ravel & Sieber (2002)** atomic database as
distributed inside XrayDB. The data are the scientific basis of every number
this software reports, and they carry their own provenance and citation
expectations that are independent of the MIT licence on this code. Cite the
underlying data source, not only this tool, in published work.

- Elam, W. T., Ravel, B. D. & Sieber, J. R. (2002). *A new atomic database for
  X-ray spectroscopic calculations.* Radiation Physics and Chemistry 63, 121–128.
  <https://doi.org/10.1016/S0969-806X(01)00227-4>
- XrayDB: <https://xraypy.github.io/XrayDB/>

The bundled materials library also contains factual composition and density
data from two public U.S. Government scientific references. Their source
records, retrieval hashes and caveats travel with every material entry.

- Hubbell, J. H. & Seltzer, S. M. *Tables of X-Ray Mass Attenuation
  Coefficients and Mass Energy-Absorption Coefficients*, NIST SRD 126.
  <https://doi.org/10.18434/T4D01F>
- Detwiler, R. S., McConn, R. J., Grimes, T. F., Upton, S. A. & Engel, E. J.
  (2021). *Compendium of Material Composition Data for Radiation Transport
  Modeling*, PNNL-15870 Rev. 2. <https://doi.org/10.2172/1782721>

## Prior work in `legacy/`

[`legacy/`](legacy/) preserves the original X-ray attenuation calculator for
attribution and reproducibility. It is archived material, not a dependency, and
**must not be used for new calculations** — see [`legacy/ARCHIVE_NOTICE.md`](legacy/ARCHIVE_NOTICE.md)
and the "Original project and publication" section of the [README](README.md).
