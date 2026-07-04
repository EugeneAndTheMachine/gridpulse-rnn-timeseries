# Data Directory

Raw data is NOT committed to git. Run `make download-data` to fetch:

## ETT-small
- Source: https://github.com/zhouhaoyi/ETDataset
- Files: ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
- 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Target: OT (oil temperature)
- Frequency: hourly (h) / 15-min (m)

## UCI Air Quality
- Source: https://archive.ics.uci.edu/dataset/360/air+quality
- 9358 hourly instances, March 2004 – February 2005
- Missing values marked as -200