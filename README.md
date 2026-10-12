# SwissPath: robust journey planning for Lausanne

SwissPath is a team project from EPFL's COM-490 course. It plans public transport journeys against an arrival deadline and a requested probability of arriving on time. The final project combines a reverse connection scan route planner, historical delay modeling, probability calibration, and an interactive Jupyter dashboard.

This repository is a portfolio copy of the final course implementation. The main notebook includes saved evaluation tables, plots, and a Lausanne route example so the work can be inspected without access to the course computing cluster.

## What it does

1. Builds transit and walking connections from Swiss public transport timetable data.
2. Trains gradient boosted tree models on historical delays and timetable, location, time, and weather features.
3. Uses validation residual distributions to estimate the probability of making each transfer and reaching the destination before the deadline.
4. Searches for timetable feasible candidate journeys, scores them, and filters them by the requested confidence level.
5. Displays routes and confidence information in a Jupyter dashboard.

The pre-trip planner uses the **reduced** delay model, which excludes live vehicle delay information. The notebook also evaluates a **full** model with live delay features for comparison; its results are not a pre-trip forecast.

## Example and evaluation

The saved Lausanne example plans a Sunday trip from **Ecublens VD, EPFL** to **Lausanne, Ouchy-Olympique** with a 15:00 arrival deadline. In that run, the notebook found 30 candidate routes; 28 passed the 50% confidence threshold, and 16 distinct routes remained after display deduplication. The latest departing displayed route left at 14:32:19 and had an estimated 91.3% chance of meeting the deadline.

On the notebook's held-out test period, the reduced model achieved **1.74 minutes MAE** and **2.91 minutes RMSE** over 5.43 million segment observations. The full model achieved 0.47 minutes MAE using additional live delay features. The notebook includes event probability calibration and a short historical trip sanity check. That sanity check is not a full end-to-end validation of journey arrival probabilities.

These are results saved from the course cluster run. They depend on the specific data period, region, and evaluation setup in the notebook.

## Repository guide

| File | Purpose |
| --- | --- |
| [`combined_delay_model_workflow.ipynb`](combined_delay_model_workflow.ipynb) | Complete data, training, evaluation, route planning, and dashboard workflow with saved outputs |
| [`journey_planner.py`](journey_planner.py) | Reverse connection scan routing and candidate journey generation |
| [`confidence_scorer.py`](confidence_scorer.py) | Delay distribution lookup and route event probability scoring |
| [`robust_journey_interface.py`](robust_journey_interface.py) | Data loading, route scoring, visualization, and dashboard helpers |
| [`robust_udfs.py`](robust_udfs.py) | Shared residual lookup utilities |

## Running the notebook

The source is preserved as a course project, so a fresh GitHub clone is suitable for reading but cannot execute the full pipeline on its own. Reproduction requires the EPFL COM-490 Spark and Trino environment, access to its Iceberg SBB tables and HDFS storage, and the `EPFL_COM490_TOKEN` environment variable. The notebook also uses Python packages including pandas, NumPy, PySpark, Trino, Matplotlib, Folium, and ipywidgets. Historical data and saved model artifacts are not included in this repository.

In that environment, open `combined_delay_model_workflow.ipynb`, review the region, timetable date, and train/validation/test periods in the configuration cells, then run the cells in order. The notebook defaults to reusing saved artifacts under the course group path; change the rebuild flags if you need to regenerate them and have access to the source data.

## Project context

This was collaborative course work. My commits in the original course repository covered the combined notebook, project structure, and example routes. The final files also include teammates' contributions, especially the routing and delay model work. The original course Git history remains in the local `final` clone; this portfolio repository presents the final project snapshot.
