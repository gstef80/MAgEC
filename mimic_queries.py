import numpy as np
import pandas as pd

demo_template = \
"""
SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
, pat.gender
, adm.admittime, adm.dischtime, adm.diagnosis
, ROUND( (CAST(adm.dischtime AS DATE) - CAST(adm.admittime AS DATE)) , 4) AS los_hospital
, ROUND( (CAST(adm.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365, 4) AS age
, adm.ethnicity, adm.ADMISSION_TYPE
, CASE when adm.deathtime between ie.intime and ie.outtime THEN 1 ELSE 0 END AS mort_icu
, DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
    ELSE 0 END AS first_hosp_stay
, ie.intime, ie.outtime
, ie.FIRST_CAREUNIT
, ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
, DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq
-- first ICU stay *for the current hospitalization*
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
    ELSE 0 END AS first_icu_stay
FROM icustays ie
INNER JOIN admissions adm
    ON ie.hadm_id = adm.hadm_id
INNER JOIN patients pat
    ON ie.subject_id = pat.subject_id
WHERE adm.has_chartevents_data = 1
ORDER BY ie.subject_id, adm.admittime, ie.intime;
"""

def demo_query():
    return demo_template


def clean_demo_df(demo, hours, drop=False):
    # no need to yell 
    df = demo.loc[(demo.age<300)].copy()
    df.loc[(df.ethnicity.str.lower().str.contains('^white')), 'ethnicity'] = 'white'
    df.loc[(df.ethnicity.str.lower().str.contains('^black')), 'ethnicity'] = 'black'
    df.loc[(df.ethnicity.str.lower().str.contains('^hisp')) | 
           (df.ethnicity.str.lower().str.contains('^latin')), 'ethnicity'] = 'hispanic'
    df.loc[(df.ethnicity.str.lower().str.contains('^asia')), 'ethnicity'] = 'asian'
    df.loc[~(df.ethnicity.str.lower().str.contains('|'.join(['white', 
                                                             'black', 
                                                             'hispanic', 
                                                             'asian']))), 'ethnicity'] = 'other'
    #----clean up
    # micu --> medical 
    # csru --> cardiac surgery recovery unit 
    # sicu --> surgical icu 
    # tsicu --> Trauma Surgical Intensive Care Unit
    # NICU --> Neonatal 
    df.loc[:, 'adult_icu'] = np.where(df['first_careunit'].isin(['PICU', 'NICU']), 0, 1)
    df.loc[:, 'gender'] = np.where(df['gender']=="M", 1, 0)
    #----drop patients with less than X hours
    df.loc[:, 'los_icu_hr'] = (df.outtime - df.intime).astype('timedelta64[h]')
    df = pd.concat([df, pd.get_dummies(df.ethnicity, prefix='eth')], 1)
    df = pd.concat([df, pd.get_dummies(df.admission_type, prefix='admType')], 1)
    if drop:
        df = df.drop(['diagnosis', 'hospstay_seq', 'los_icu','icustay_seq', 'admittime', 
                      'dischtime','los_hospital', 'intime', 'outtime', 'ethnicity', 
                      'admission_type', 'first_careunit'], 1) 
    return df.loc[(df.los_icu_hr >= hours)]


lab_template = \
"""
WITH pvt AS (
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime
  , CASE
        when le.itemid = 50868 then 'ANION GAP'
        when le.itemid = 50862 then 'ALBUMIN'
        when le.itemid = 50882 then 'BICARBONATE'
        when le.itemid = 50885 then 'BILIRUBIN'
        when le.itemid = 50912 then 'CREATININE'
        when le.itemid = 50806 then 'CHLORIDE'
        when le.itemid = 50902 then 'CHLORIDE'
        when le.itemid = 50809 then 'GLUCOSE'
        when le.itemid = 50931 then 'GLUCOSE'
        when le.itemid = 50810 then 'HEMATOCRIT'
        when le.itemid = 51221 then 'HEMATOCRIT'
        when le.itemid = 50811 then 'HEMOGLOBIN'
        when le.itemid = 51222 then 'HEMOGLOBIN'
        when le.itemid = 50813 then 'LACTATE'
        when le.itemid = 50960 then 'MAGNESIUM'
        when le.itemid = 50970 then 'PHOSPHATE'
        when le.itemid = 51265 then 'PLATELET'
        when le.itemid = 50822 then 'POTASSIUM'
        when le.itemid = 50971 then 'POTASSIUM'
        when le.itemid = 51275 then 'PTT'
        when le.itemid = 51237 then 'INR'
        when le.itemid = 51274 then 'PT'
        when le.itemid = 50824 then 'SODIUM'
        when le.itemid = 50983 then 'SODIUM'
        when le.itemid = 51006 then 'BUN'
        when le.itemid = 51300 then 'WBC'
        when le.itemid = 51301 then 'WBC'
      ELSE null
      END AS label
  , -- add in some sanity checks on the values
    -- the where clause below requires all valuenum to be > 0, 
    -- so these are only upper limit checks
    CASE
      when le.itemid = 50862 and le.valuenum >    10 then null -- g/dL 'ALBUMIN'
      when le.itemid = 50868 and le.valuenum > 10000 then null -- mEq/L 'ANION GAP'
      when le.itemid = 50882 and le.valuenum > 10000 then null -- mEq/L 'BICARBONATE'
      when le.itemid = 50885 and le.valuenum >   150 then null -- mg/dL 'BILIRUBIN'
      when le.itemid = 50806 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
      when le.itemid = 50902 and le.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
      when le.itemid = 50912 and le.valuenum >   150 then null -- mg/dL 'CREATININE'
      when le.itemid = 50809 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
      when le.itemid = 50931 and le.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
      when le.itemid = 50810 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
      when le.itemid = 51221 and le.valuenum >   100 then null -- % 'HEMATOCRIT'
      when le.itemid = 50811 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
      when le.itemid = 51222 and le.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
      when le.itemid = 50813 and le.valuenum >    50 then null -- mmol/L 'LACTATE'
      when le.itemid = 50960 and le.valuenum >    60 then null -- mmol/L 'MAGNESIUM'
      when le.itemid = 50970 and le.valuenum >    60 then null -- mg/dL 'PHOSPHATE'
      when le.itemid = 51265 and le.valuenum > 10000 then null -- K/uL 'PLATELET'
      when le.itemid = 50822 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
      when le.itemid = 50971 and le.valuenum >    30 then null -- mEq/L 'POTASSIUM'
      when le.itemid = 51275 and le.valuenum >   150 then null -- sec 'PTT'
      when le.itemid = 51237 and le.valuenum >    50 then null -- 'INR'
      when le.itemid = 51274 and le.valuenum >   150 then null -- sec 'PT'
      when le.itemid = 50824 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
      when le.itemid = 50983 and le.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
      when le.itemid = 51006 and le.valuenum >   300 then null -- 'BUN'
      when le.itemid = 51300 and le.valuenum >  1000 then null -- 'WBC'
      when le.itemid = 51301 and le.valuenum >  1000 then null -- 'WBC'
    ELSE le.valuenum
    END AS valuenum
  FROM icustays ie

  LEFT JOIN labevents le
    ON le.subject_id = ie.subject_id 
    AND le.hadm_id = ie.hadm_id
    -- TODO: they are using lab times 6 hours before the start of the 
    -- ICU stay. 
    AND le.charttime between (ie.intime - interval '6' hour) 
    AND (ie.intime + interval '{}' hour) -- extract the lab events in the first X hours
    AND le.itemid IN
    (
      -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
      50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
      50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
      50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
      50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
      50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
      50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
      50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
      50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
      50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
      51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
      50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
      51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
      50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
      50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
      50960, -- MAGNESIUM | CHEMISTRY | BLOOD | 664191
      50970, -- PHOSPHATE | CHEMISTRY | BLOOD | 590524
      51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
      50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
      50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
      51275, -- PTT | HEMATOLOGY | BLOOD | 474937
      51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
      51274, -- PT | HEMATOLOGY | BLOOD | 469090
      50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
      50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
      51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
      51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
      51300  -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
    )
    AND le.valuenum IS NOT null 
    AND le.valuenum > 0 -- lab values cannot be 0 and cannot be negative
    
    LEFT JOIN admissions ad
    ON ie.subject_id = ad.subject_id
    AND ie.hadm_id = ad.hadm_id   
),
ranked AS (
SELECT pvt.*, DENSE_RANK() OVER (PARTITION BY 
    pvt.subject_id, pvt.hadm_id,pvt.icustay_id,pvt.label ORDER BY pvt.charttime) as drank
FROM pvt
)
SELECT r.subject_id, r.hadm_id, r.icustay_id, date(r.charttime) as lab_date
  , max(case when label = 'ANION GAP' then valuenum else null end) as ANIONGAP
  , max(case when label = 'ALBUMIN' then valuenum else null end) as ALBUMIN
  , max(case when label = 'BICARBONATE' then valuenum else null end) as BICARBONATE
  , max(case when label = 'BILIRUBIN' then valuenum else null end) as BILIRUBIN
  , max(case when label = 'CREATININE' then valuenum else null end) as CREATININE
  , max(case when label = 'CHLORIDE' then valuenum else null end) as CHLORIDE
  , max(case when label = 'GLUCOSE' then valuenum else null end) as GLUCOSE
  , max(case when label = 'HEMATOCRIT' then valuenum else null end) as HEMATOCRIT
  , max(case when label = 'HEMOGLOBIN' then valuenum else null end) as HEMOGLOBIN
  , max(case when label = 'LACTATE' then valuenum else null end) as LACTATE
  , max(case when label = 'MAGNESIUM' then valuenum else null end) as MAGNESIUM
  , max(case when label = 'PHOSPHATE' then valuenum else null end) as PHOSPHATE
  , max(case when label = 'PLATELET' then valuenum else null end) as PLATELET
  , max(case when label = 'POTASSIUM' then valuenum else null end) as POTASSIUM
  , max(case when label = 'PTT' then valuenum else null end) as PTT
  , max(case when label = 'INR' then valuenum else null end) as INR
  , max(case when label = 'PT' then valuenum else null end) as PT
  , max(case when label = 'SODIUM' then valuenum else null end) as SODIUM
  , max(case when label = 'BUN' then valuenum else null end) as BUN
  , max(case when label = 'WBC' then valuenum else null end) as WBC

FROM ranked r
WHERE r.drank = 1
GROUP BY r.subject_id, r.hadm_id, r.icustay_id,  r.drank, lab_date
ORDER BY r.subject_id, r.hadm_id, r.icustay_id,  r.drank, lab_date;
"""

def lab_query(hours):
    # This query does the following:
    #   it extracts the lab events in the first X hours 
    #   it labels the lab items and cleans up their values 
    #   it will create a set of lab values 
    #   X hours. 
    return lab_template.format(hours)

vitals_template = \
"""
-- This query pivots the vital signs for the first X hours of a patient's stay
-- Vital signs include heart rate, blood pressure, respiration rate, and temperature
SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.lab_date
-- Easier names
, min(case when VitalID = 1 then valuenum else null end) as HeartRate_Min
, max(case when VitalID = 1 then valuenum else null end) as HeartRate_Max
, avg(case when VitalID = 1 then valuenum else null end) as HeartRate_Mean
, min(case when VitalID = 2 then valuenum else null end) as SysBP_Min
, max(case when VitalID = 2 then valuenum else null end) as SysBP_Max
, avg(case when VitalID = 2 then valuenum else null end) as SysBP_Mean
, min(case when VitalID = 3 then valuenum else null end) as DiasBP_Min
, max(case when VitalID = 3 then valuenum else null end) as DiasBP_Max
, avg(case when VitalID = 3 then valuenum else null end) as DiasBP_Mean
, min(case when VitalID = 4 then valuenum else null end) as MeanBP_Min
, max(case when VitalID = 4 then valuenum else null end) as MeanBP_Max
, avg(case when VitalID = 4 then valuenum else null end) as MeanBP_Mean
, min(case when VitalID = 5 then valuenum else null end) as RespRate_Min
, max(case when VitalID = 5 then valuenum else null end) as RespRate_Max
, avg(case when VitalID = 5 then valuenum else null end) as RespRate_Mean
, min(case when VitalID = 6 then valuenum else null end) as TempC_Min
, max(case when VitalID = 6 then valuenum else null end) as TempC_Max
, avg(case when VitalID = 6 then valuenum else null end) as TempC_Mean
, min(case when VitalID = 7 then valuenum else null end) as SpO2_Min
, max(case when VitalID = 7 then valuenum else null end) as SpO2_Max
, avg(case when VitalID = 7 then valuenum else null end) as SpO2_Mean
, min(case when VitalID = 8 then valuenum else null end) as Glucose_Min
, max(case when VitalID = 8 then valuenum else null end) as Glucose_Max
, avg(case when VitalID = 8 then valuenum else null end) as Glucose_Mean

FROM  (
  select ie.subject_id, ie.hadm_id, ie.icustay_id, date(ce.charttime) as lab_date
  , case
    when itemid in (211,220045) and valuenum > 0 and valuenum < 300 then 1 -- HeartRate
    when itemid in (51,442,455,6701,220179,220050) and valuenum > 0 and valuenum < 400 then 2 -- SysBP
    when itemid in (8368,8440,8441,8555,220180,220051) and valuenum > 0 and valuenum < 300 then 3 -- DiasBP
    when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then 4 -- MeanBP
    when itemid in (615,618,220210,224690) and valuenum > 0 and valuenum < 70 then 5 -- RespRate
    when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then 6 -- TempF, converted to degC in valuenum call
    when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then 6 -- TempC
    when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then 7 -- SpO2
    when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then 8 -- Glucose

    else null end as VitalID
      -- convert F to C
  , case when itemid in (223761,678) then (valuenum-32)/1.8 else valuenum end as valuenum

  from icustays ie
  left join chartevents ce
  on ie.subject_id = ce.subject_id and ie.hadm_id = ce.hadm_id and ie.icustay_id = ce.icustay_id
  and ce.charttime between ie.intime and ie.intime + interval '{}' hour -- extraxt first X hours
  -- exclude rows marked as error
  and ce.error IS DISTINCT FROM 1
  where ce.itemid in
  (
  -- HEART RATE
  211, --"Heart Rate"
  220045, --"Heart Rate"

  -- Systolic/diastolic

  51, -- Arterial BP [Systolic]
  442, -- Manual BP [Systolic]
  455, -- NBP [Systolic]
  6701, -- Arterial BP #2 [Systolic]
  220179, -- Non Invasive Blood Pressure systolic
  220050, -- Arterial Blood Pressure systolic
  8368, -- Arterial BP [Diastolic]
  8440, -- Manual BP [Diastolic]
  8441, -- NBP [Diastolic]
  8555, -- Arterial BP #2 [Diastolic]
  220180, -- Non Invasive Blood Pressure diastolic
  220051, -- Arterial Blood Pressure diastolic

  -- MEAN ARTERIAL PRESSURE
  456, -- "NBP Mean"
  52, -- "Arterial BP Mean"
  6702, -- Arterial BP Mean #2
  443, -- Manual BP Mean(calc)
  220052, -- "Arterial Blood Pressure mean"
  220181, -- "Non Invasive Blood Pressure mean"
  225312, -- "ART BP mean"

  -- RESPIRATORY RATE
  618, -- Respiratory Rate
  615, -- Resp Rate (Total)
  220210, -- Respiratory Rate
  224690, -- Respiratory Rate (Total)

  -- SPO2, peripheral
  646, 220277,

  -- GLUCOSE, both lab and fingerstick
  807, -- Fingerstick Glucose
  811, -- Glucose (70-105)
  1529, -- Glucose
  3745, -- BloodGlucose
  3744, -- Blood Glucose
  225664, -- Glucose finger stick
  220621, -- Glucose (serum)
  226537, -- Glucose (whole blood)

  -- TEMPERATURE
  223762, -- "Temperature Celsius"
  676, -- "Temperature C"
  223761, -- "Temperature Fahrenheit"
  678 -- "Temperature F"

  )
) pvt
group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.lab_date
order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.lab_date;
"""

def vitals_query(hours):
    return vitals_template.format(hours)


ventilator_template = \
"""
-- This query extracts the duration of mechanical ventilation
-- The main goal of the query is to aggregate sequential ventilator settings
-- into single mechanical ventilation "events". The start and end time of these
-- events can then be used for various purposes: calculating the total duration
-- of mechanical ventilation, cross-checking values (e.g. PaO2:FiO2 on vent), etc

-- The query's logic is roughly:
--    1) The presence of a mechanical ventilation setting starts a new ventilation event
--    2) Any instance of a setting in the next 8 hours continues the event
--    3) Certain elements end the current ventilation event
--        a) documented extubation ends the current ventilation
--        b) initiation of non-invasive vent and/or oxygen ends the current vent
-- The ventilation events are numbered consecutively by the `num` column.

with ventsettings as
(
    select
      icustay_id, charttime
      -- case statement determining whether it is an instance of mech vent
      , max(
        case
          when itemid is null or value is null then 0 -- can't have null values
          when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
          when itemid = 223848 and value != 'Other' THEN 1
          when itemid = 223849 then 1 -- ventilator mode
          when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
          when itemid in
            (
            445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
            , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
            , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
            , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
            , 543 -- PlateauPressure
            , 5865,5866,224707,224709,224705,224706 -- APRV pressure
            , 60,437,505,506,686,220339,224700 -- PEEP
            , 3459 -- high pressure relief
            , 501,502,503,224702 -- PCV
            , 223,667,668,669,670,671,672 -- TCPCV
            , 224701 -- PSVlevel
            )
            THEN 1
          else 0
        end
        ) as MechVent
        , max(
          case
            -- initiation of oxygen therapy indicates the ventilation has ended
            when itemid = 226732 and value in
            (
              'Nasal cannula', -- 153714 observations
              'Face tent', -- 24601 observations
              'Aerosol-cool', -- 24560 observations
              'Trach mask ', -- 16435 observations
              'High flow neb', -- 10785 observations
              'Non-rebreather', -- 5182 observations
              'Venti mask ', -- 1947 observations
              'Medium conc mask ', -- 1888 observations
              'T-piece', -- 1135 observations
              'High flow nasal cannula', -- 925 observations
              'Ultrasonic neb', -- 9 observations
              'Vapomist' -- 3 observations
            ) then 1
            when itemid = 467 and value in
            (
              'Cannula', -- 278252 observations
              'Nasal Cannula', -- 248299 observations
              -- 'None', -- 95498 observations
              'Face Tent', -- 35766 observations
              'Aerosol-Cool', -- 33919 observations
              'Trach Mask', -- 32655 observations
              'Hi Flow Neb', -- 14070 observations
              'Non-Rebreather', -- 10856 observations
              'Venti Mask', -- 4279 observations
              'Medium Conc Mask', -- 2114 observations
              'Vapotherm', -- 1655 observations
              'T-Piece', -- 779 observations
              'Hood', -- 670 observations
              'Hut', -- 150 observations
              'TranstrachealCat', -- 78 observations
              'Heated Neb', -- 37 observations
              'Ultrasonic Neb' -- 2 observations
            ) then 1
          else 0
          end
        ) as OxygenTherapy
        , max(
          case when itemid is null or value is null then 0
            -- extubated indicates ventilation event has ended
            when itemid = 640 and value = 'Extubated' then 1
            when itemid = 640 and value = 'Self Extubation' then 1
          else 0
          end
          )
          as Extubated
        , max(
          case when itemid is null or value is null then 0
            when itemid = 640 and value = 'Self Extubation' then 1
          else 0
          end
          )
          as SelfExtubated
    from chartevents ce
    where ce.value is not null
    -- exclude rows marked as error
    and ce.error IS DISTINCT FROM 1
    and itemid in
    (
        -- the below are settings used to indicate ventilation
          720, 223849 -- vent mode
        , 223848 -- vent type
        , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
        , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
        , 218,436,535,444,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
        , 543 -- PlateauPressure
        , 5865,5866,224707,224709,224705,224706 -- APRV pressure
        , 60,437,505,506,686,220339,224700 -- PEEP
        , 3459 -- high pressure relief
        , 501,502,503,224702 -- PCV
        , 223,667,668,669,670,671,672 -- TCPCV
        , 224701 -- PSVlevel

        -- the below are settings used to indicate extubation
        , 640 -- extubated

        -- the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
        , 468 -- O2 Delivery Device#2
        , 469 -- O2 Delivery Mode
        , 470 -- O2 Flow (lpm)
        , 471 -- O2 Flow (lpm) #2
        , 227287 -- O2 Flow (additional cannula)
        , 226732 -- O2 Delivery Device(s)
        , 223834 -- O2 Flow

        -- used in both oxygen + vent calculation
        , 467 -- O2 Delivery Device
    )
    group by icustay_id, charttime
    UNION
    -- add in the extubation flags from procedureevents_mv
    -- note that we only need the start time for the extubation
    -- (extubation is always charted as ending 1 minute after it started)
    select
      icustay_id, starttime as charttime
      , 0 as MechVent
      , 0 as OxygenTherapy
      , 1 as Extubated
      , case when itemid = 225468 then 1 else 0 end as SelfExtubated
    from procedureevents_mv
    where itemid in
    (
      227194 -- "Extubation"
    , 225468 -- "Unplanned Extubation (patient-initiated)"
    , 225477 -- "Unplanned Extubation (non-patient initiated)"
    )
)
, vd0 as
(
  select
    icustay_id
    -- this carries over the previous charttime which had a mechanical ventilation event
    , case
        when MechVent=1 then
          LAG(CHARTTIME, 1) OVER (partition by icustay_id, MechVent order by charttime)
        else null
      end as charttime_lag
    , charttime
    , MechVent
    , OxygenTherapy
    , Extubated
    , SelfExtubated
  from ventsettings
)
, vd1 as
(
  select
      icustay_id
      , charttime_lag
      , charttime
      , MechVent
      , OxygenTherapy
      , Extubated
      , SelfExtubated

      -- if this is a mechanical ventilation event, we calculate the time since the last event
      , case
          -- if the current observation indicates mechanical ventilation is present
          -- calculate the time since the last vent event
          when MechVent=1 then
            CHARTTIME - charttime_lag
          else null
        end as ventduration

      , LAG(Extubated,1)
      OVER
      (
      partition by icustay_id, case when MechVent=1 or Extubated=1 then 1 else 0 end
      order by charttime
      ) as ExtubatedLag

      -- now we determine if the current mech vent event is a "new", i.e. they've just been intubated
      , case
        -- if there is an extubation flag, we mark any subsequent ventilation as a new ventilation event
          --when Extubated = 1 then 0 -- extubation is *not* a new ventilation event, the *subsequent* row is
          when
            LAG(Extubated,1)
            OVER
            (
            partition by icustay_id, case when MechVent=1 or Extubated=1 then 1 else 0 end
            order by charttime
            )
            = 1 then 1
          -- if patient has initiated oxygen therapy, and is not currently vented, start a newvent
          when MechVent = 0 and OxygenTherapy = 1 then 1
            -- if there is less than 8 hours between vent settings, we do not treat this as a new ventilation event
          when (CHARTTIME - charttime_lag) > interval '8' hour
            then 1
        else 0
        end as newvent
  -- use the staging table with only vent settings from chart events
  FROM vd0 ventsettings
)
, vd2 as
(
  select vd1.*
  -- create a cumulative sum of the instances of new ventilation
  -- this results in a monotonic integer assigned to each instance of ventilation
  , case when MechVent=1 or Extubated = 1 then
      SUM( newvent )
      OVER ( partition by icustay_id order by charttime )
    else null end
    as ventnum
  --- now we convert CHARTTIME of ventilator settings into durations
  from vd1
)
-- create the durations for each mechanical ventilation instance
select icustay_id
  -- regenerate ventnum so it's sequential
  , ROW_NUMBER() over (partition by icustay_id order by ventnum) as ventnum
  , min(charttime) as mv_start
  , max(charttime) as mv_end
  , extract(epoch from max(charttime)-min(charttime))/60/60 AS mv_hours
from vd2
group by icustay_id, ventnum
having min(charttime) != max(charttime)
-- patient had to be mechanically ventilated at least once
-- i.e. max(mechvent) should be 1
-- this excludes a frequent situation of NIV/oxygen before intub
-- in these cases, ventnum=0 and max(mechvent)=0, so they are ignored
and max(mechvent) = 1
order by icustay_id, ventnum;
"""

def vent_query():
    return ventilator_template

