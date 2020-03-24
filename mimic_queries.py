import numpy as np
import pandas as pd


def services_query():
    return "select subject_id, hadm_id, transfertime, prev_service, curr_service from services;"


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
    """
    SQL query for extracting demographics from mimics data
    """
    return demo_template


def clean_demo_df(demo, hours, drop=True):
    """
    INPUT: mimics demographics pandas dataframe
    OUTPUT: pandas dataframe with patients who stayed in ICU for at least 'hours' hours
    """
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
    # df = pd.concat([df, pd.get_dummies(df.ethnicity, prefix='eth')], 1)
    # df = pd.concat([df, pd.get_dummies(df.admission_type, prefix='admType')], 1)
    if drop:
        df = df.drop(['hospstay_seq', 'icustay_seq'], 1) 
    return df.loc[(df.los_icu_hr >= hours)]


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
    """
    From 'concepts' in MIMICIII
    This query extracts the duration of mechanical ventilation
    The main goal of the query is to aggregate sequential ventilator settings
    into single mechanical ventilation "events". The start and end time of these
    events can then be used for various purposes: calculating the total duration
    of mechanical ventilation, cross-checking values (e.g. PaO2:FiO2 on vent), etc

    The query's logic is roughly:
      1) The presence of a mechanical ventilation setting starts a new ventilation event
      2) Any instance of a setting in the next 8 hours continues the event
      3) Certain elements end the current ventilation event
          a) documented extubation ends the current ventilation
          b) initiation of non-invasive vent and/or oxygen ends the current vent
   The ventilation events are numbered consecutively by the `num` column.
    """
    return ventilator_template


comob_template = \
"""
-- This code calculates the Elixhauser comorbidities as defined in Quan et. al 2009:
-- Quan, Hude, et al. "Coding algorithms for defining comorbidities in
-- ICD-9-CM and ICD-10 administrative data." Medical care (2005): 1130-1139.
--  https://www.ncbi.nlm.nih.gov/pubmed/16224307

-- Quan defined an "Enhanced ICD-9" coding scheme for deriving Elixhauser
-- comorbidities from ICD-9 billing codes. This script implements that calculation.

-- The logic of the code is roughly that, if the comorbidity lists a length 3
-- ICD-9 code (e.g. 585), then we only require a match on the first 3 characters.

-- This code derives each comorbidity as follows:
--  1) ICD9_CODE is directly compared to 5 character codes
--  2) The first 4 characters of ICD9_CODE are compared to 4 character codes
--  3) The first 3 characters of ICD9_CODE are compared to 3 character codes


with icd as
(
  select hadm_id, seq_num, icd9_code
  from diagnoses_icd
  where seq_num != 1 -- we do not include the primary icd-9 code
)
, eliflg as
(
select hadm_id, seq_num, icd9_code
, CASE
  when icd9_code in ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4254','4255','4257','4258','4259') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('428') then 1
  else 0 end as CHF       /* Congestive heart failure */

, CASE
  when icd9_code in ('42613','42610','42612','99601','99604') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4260','4267','4269','4270','4271','4272','4273','4274','4276','4278','4279','7850','V450','V533') then 1
  else 0 end as ARRHY

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('0932','7463','7464','7465','7466','V422','V433') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('394','395','396','397','424') then 1
  else 0 end as VALVE     /* Valvular disease */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4150','4151','4170','4178','4179') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('416') then 1
  else 0 end as PULMCIRC  /* Pulmonary circulation disorder */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('0930','4373','4431','4432','4438','4439','4471','5571','5579','V434') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('440','441') then 1
  else 0 end as PERIVASC  /* Peripheral vascular disorder */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('401') then 1
  else 0 end as HTN       /* Hypertension, uncomplicated */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('402','403','404','405') then 1
  else 0 end as HTNCX     /* Hypertension, complicated */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('3341','3440','3441','3442','3443','3444','3445','3446','3449') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('342','343') then 1
  else 0 end as PARA      /* Paralysis */

, CASE
  when icd9_code in ('33392') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('3319','3320','3321','3334','3335','3362','3481','3483','7803','7843') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('334','335','340','341','345') then 1
  else 0 end as NEURO     /* Other neurological */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('4168','4169','5064','5081','5088') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('490','491','492','493','494','495','496','500','501','502','503','504','505') then 1
  else 0 end as CHRNLUNG  /* Chronic pulmonary disease */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2500','2501','2502','2503') then 1
  else 0 end as DM        /* Diabetes w/o chronic complications*/

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2504','2505','2506','2507','2508','2509') then 1
  else 0 end as DMCX      /* Diabetes w/ chronic complications */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2409','2461','2468') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('243','244') then 1
  else 0 end as HYPOTHY   /* Hypothyroidism */

, CASE
  when icd9_code in ('40301','40311','40391','40402','40403','40412','40413','40492','40493') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('5880','V420','V451') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('585','586','V56') then 1
  else 0 end as RENLFAIL  /* Renal failure */

, CASE
  when icd9_code in ('07022','07023','07032','07033','07044','07054') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('0706','0709','4560','4561','4562','5722','5723','5724','5728','5733','5734','5738','5739','V427') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('570','571') then 1
  else 0 end as LIVER     /* Liver disease */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('5317','5319','5327','5329','5337','5339','5347','5349') then 1
  else 0 end as ULCER     /* Chronic Peptic ulcer disease (includes bleeding only if obstruction is also present) */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('042','043','044') then 1
  else 0 end as AIDS      /* HIV and AIDS */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2030','2386') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('200','201','202') then 1
  else 0 end as LYMPH     /* Lymphoma */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in ('196','197','198','199') then 1
  else 0 end as METS      /* Metastatic cancer */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 3) in
  (
     '140','141','142','143','144','145','146','147','148','149','150','151','152'
    ,'153','154','155','156','157','158','159','160','161','162','163','164','165'
    ,'166','167','168','169','170','171','172','174','175','176','177','178','179'
    ,'180','181','182','183','184','185','186','187','188','189','190','191','192'
    ,'193','194','195'
  ) then 1
  else 0 end as TUMOR     /* Solid tumor without metastasis */

, CASE
  when icd9_code in ('72889','72930') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('7010','7100','7101','7102','7103','7104','7108','7109','7112','7193','7285') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('446','714','720','725') then 1
  else 0 end as ARTH              /* Rheumatoid arthritis/collagen vascular diseases */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2871','2873','2874','2875') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('286') then 1
  else 0 end as COAG      /* Coagulation deficiency */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2780') then 1
  else 0 end as OBESE     /* Obesity      */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('7832','7994') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('260','261','262','263') then 1
  else 0 end as WGHTLOSS  /* Weight loss */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2536') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('276') then 1
  else 0 end as LYTES     /* Fluid and electrolyte disorders */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2800') then 1
  else 0 end as BLDLOSS   /* Blood loss anemia */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2801','2808','2809') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('281') then 1
  else 0 end as ANEMDEF  /* Deficiency anemias */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2652','2911','2912','2913','2915','2918','2919','3030','3039','3050','3575','4255','5353','5710','5711','5712','5713','V113') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('980') then 1
  else 0 end as ALCOHOL /* Alcohol abuse */

, CASE
  when icd9_code in ('V6542') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('3052','3053','3054','3055','3056','3057','3058','3059') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('292','304') then 1
  else 0 end as DRUG /* Drug abuse */

, CASE
  when icd9_code in ('29604','29614','29644','29654') then 1
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2938') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('295','297','298') then 1
  else 0 end as PSYCH /* Psychoses */

, CASE
  when SUBSTRING(icd9_code FROM 1 for 4) in ('2962','2963','2965','3004') then 1
  when SUBSTRING(icd9_code FROM 1 for 3) in ('309','311') then 1
  else 0 end as DEPRESS  /* Depression */
from icd
)
-- collapse the icd9_code specific flags into hadm_id specific flags
-- this groups comorbidities together for a single patient admission
, eligrp as
(
  select hadm_id
  , max(chf) as chf
  , max(arrhy) as arrhy
  , max(valve) as valve
  , max(pulmcirc) as pulmcirc
  , max(perivasc) as perivasc
  , max(htn) as htn
  , max(htncx) as htncx
  , max(para) as para
  , max(neuro) as neuro
  , max(chrnlung) as chrnlung
  , max(dm) as dm
  , max(dmcx) as dmcx
  , max(hypothy) as hypothy
  , max(renlfail) as renlfail
  , max(liver) as liver
  , max(ulcer) as ulcer
  , max(aids) as aids
  , max(lymph) as lymph
  , max(mets) as mets
  , max(tumor) as tumor
  , max(arth) as arth
  , max(coag) as coag
  , max(obese) as obese
  , max(wghtloss) as wghtloss
  , max(lytes) as lytes
  , max(bldloss) as bldloss
  , max(anemdef) as anemdef
  , max(alcohol) as alcohol
  , max(drug) as drug
  , max(psych) as psych
  , max(depress) as depress
from eliflg
group by hadm_id
)
-- now merge these flags together to define elixhauser
-- most are straightforward.. but hypertension flags are a bit more complicated


select adm.hadm_id
, chf as CONGESTIVE_HEART_FAILURE
, arrhy as CARDIAC_ARRHYTHMIAS
, valve as VALVULAR_DISEASE
, pulmcirc as PULMONARY_CIRCULATION
, perivasc as PERIPHERAL_VASCULAR
-- we combine "htn" and "htncx" into "HYPERTENSION"
, case
    when htn = 1 then 1
    when htncx = 1 then 1
  else 0 end as HYPERTENSION
, para as PARALYSIS
, neuro as OTHER_NEUROLOGICAL
, chrnlung as CHRONIC_PULMONARY
-- only the more severe comorbidity (complicated diabetes) is kept
, case
    when dmcx = 1 then 0
    when dm = 1 then 1
  else 0 end as DIABETES_UNCOMPLICATED
, dmcx as DIABETES_COMPLICATED
, hypothy as HYPOTHYROIDISM
, renlfail as RENAL_FAILURE
, liver as LIVER_DISEASE
, ulcer as PEPTIC_ULCER
, aids as AIDS
, lymph as LYMPHOMA
, mets as METASTATIC_CANCER
-- only the more severe comorbidity (metastatic cancer) is kept
, case
    when mets = 1 then 0
    when tumor = 1 then 1
  else 0 end as SOLID_TUMOR
, arth as RHEUMATOID_ARTHRITIS
, coag as COAGULOPATHY
, obese as OBESITY
, wghtloss as WEIGHT_LOSS
, lytes as FLUID_ELECTROLYTE
, bldloss as BLOOD_LOSS_ANEMIA
, anemdef as DEFICIENCY_ANEMIAS
, alcohol as ALCOHOL_ABUSE
, drug as DRUG_ABUSE
, psych as PSYCHOSES
, depress as DEPRESSION

from admissions adm
left join eligrp eli
  on adm.hadm_id = eli.hadm_id
order by adm.hadm_id;
"""


def comob_query(): 
    """
    SQL query for mimicsiii data computing Elixhauser comorbidities as defined in Quan et. al 2009
    """
    return comob_template


vital_and_labs_template = \
"""
-- first positional argument is Window plus Lead, second is Lead
WITH mv_labs AS (
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime, le.itemid, le.valuenum
  , CASE 
        when mv.ventnum is null then mv.transfertime + interval '{0}' hour -- proxy for event_time
        ELSE mv.mv_start  -- event_time is the start of mechanical ventilation
        END as event_time
  , CASE 
        when mv.ventnum is null then 0 else 1
        END as ventilated
  , mv.mv_start
  
  FROM icustays ie
  
  INNER JOIN mv_users mv
    ON ie.subject_id = mv.subject_id
    AND ie.hadm_id = mv.hadm_id
    AND ie.icustay_id = mv.icustay_id
  
  LEFT JOIN labevents le
    ON le.subject_id = ie.subject_id 
    AND le.hadm_id = ie.hadm_id
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
),
pvt AS (
  SELECT mv.subject_id, mv.hadm_id, mv.icustay_id, mv.event_time, mv.charttime, mv.ventilated, mv.mv_start
  , CASE
        when mv.itemid = 50868 then 'ANION GAP'
        when mv.itemid = 50862 then 'ALBUMIN'
        when mv.itemid = 50882 then 'BICARBONATE'
        when mv.itemid = 50885 then 'BILIRUBIN'
        when mv.itemid = 50912 then 'CREATININE'
        when mv.itemid = 50806 then 'CHLORIDE'
        when mv.itemid = 50902 then 'CHLORIDE'
        when mv.itemid = 50809 then 'GLUCOSE'
        when mv.itemid = 50931 then 'GLUCOSE'
        when mv.itemid = 50810 then 'HEMATOCRIT'
        when mv.itemid = 51221 then 'HEMATOCRIT'
        when mv.itemid = 50811 then 'HEMOGLOBIN'
        when mv.itemid = 51222 then 'HEMOGLOBIN'
        when mv.itemid = 50813 then 'LACTATE'
        when mv.itemid = 50960 then 'MAGNESIUM'
        when mv.itemid = 50970 then 'PHOSPHATE'
        when mv.itemid = 51265 then 'PLATELET'
        when mv.itemid = 50822 then 'POTASSIUM'
        when mv.itemid = 50971 then 'POTASSIUM'
        when mv.itemid = 51275 then 'PTT'
        when mv.itemid = 51237 then 'INR'
        when mv.itemid = 51274 then 'PT'
        when mv.itemid = 50824 then 'SODIUM'
        when mv.itemid = 50983 then 'SODIUM'
        when mv.itemid = 51006 then 'BUN'
        when mv.itemid = 51300 then 'WBC'
        when mv.itemid = 51301 then 'WBC'
      ELSE null
      END AS label
  , -- add in some sanity checks on the values
    -- the where clause below requires all valuenum to be > 0, 
    -- so these are only upper limit checks
    CASE
      when mv.itemid = 50862 and mv.valuenum >    10 then null -- g/dL 'ALBUMIN'
      when mv.itemid = 50868 and mv.valuenum > 10000 then null -- mEq/L 'ANION GAP'
      when mv.itemid = 50882 and mv.valuenum > 10000 then null -- mEq/L 'BICARBONATE'
      when mv.itemid = 50885 and mv.valuenum >   150 then null -- mg/dL 'BILIRUBIN'
      when mv.itemid = 50806 and mv.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
      when mv.itemid = 50902 and mv.valuenum > 10000 then null -- mEq/L 'CHLORIDE'
      when mv.itemid = 50912 and mv.valuenum >   150 then null -- mg/dL 'CREATININE'
      when mv.itemid = 50809 and mv.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
      when mv.itemid = 50931 and mv.valuenum > 10000 then null -- mg/dL 'GLUCOSE'
      when mv.itemid = 50810 and mv.valuenum >   100 then null -- percent 'HEMATOCRIT'
      when mv.itemid = 51221 and mv.valuenum >   100 then null -- percent 'HEMATOCRIT'
      when mv.itemid = 50811 and mv.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
      when mv.itemid = 51222 and mv.valuenum >    50 then null -- g/dL 'HEMOGLOBIN'
      when mv.itemid = 50813 and mv.valuenum >    50 then null -- mmol/L 'LACTATE'
      when mv.itemid = 50960 and mv.valuenum >    60 then null -- mmol/L 'MAGNESIUM'
      when mv.itemid = 50970 and mv.valuenum >    60 then null -- mg/dL 'PHOSPHATE'
      when mv.itemid = 51265 and mv.valuenum > 10000 then null -- K/uL 'PLATELET'
      when mv.itemid = 50822 and mv.valuenum >    30 then null -- mEq/L 'POTASSIUM'
      when mv.itemid = 50971 and mv.valuenum >    30 then null -- mEq/L 'POTASSIUM'
      when mv.itemid = 51275 and mv.valuenum >   150 then null -- sec 'PTT'
      when mv.itemid = 51237 and mv.valuenum >    50 then null -- 'INR'
      when mv.itemid = 51274 and mv.valuenum >   150 then null -- sec 'PT'
      when mv.itemid = 50824 and mv.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
      when mv.itemid = 50983 and mv.valuenum >   200 then null -- mEq/L == mmol/L 'SODIUM'
      when mv.itemid = 51006 and mv.valuenum >   300 then null -- 'BUN'
      when mv.itemid = 51300 and mv.valuenum >  1000 then null -- 'WBC'
      when mv.itemid = 51301 and mv.valuenum >  1000 then null -- 'WBC'
    ELSE mv.valuenum
    END AS valuenum
  FROM mv_labs mv
),
ranked AS (
SELECT pvt.*, DENSE_RANK() OVER (PARTITION BY 
    pvt.subject_id, pvt.hadm_id,pvt.icustay_id,pvt.label ORDER BY pvt.charttime) as drank
FROM pvt
),
labs AS (
SELECT r.subject_id, r.hadm_id, r.icustay_id
  , min(r.event_time) as event_time
  , min(r.ventilated) as ventilated
  , min(r.mv_start) as mv_start
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
AND r.charttime < event_time - interval '{1}' hour
GROUP BY r.subject_id, r.hadm_id, r.icustay_id,  r.drank
)
SELECT pvt2.subject_id, pvt2.hadm_id, pvt2.icustay_id, pvt2.timepoint::integer as timepoint
  , min(pvt2.event_time) as event_time
  , min(pvt2.ventilated) as ventilated
  , min(pvt2.mv_start) as mv_start
  , min(pvt2.ANIONGAP) as ANIONGAP
  , min(pvt2.ALBUMIN) as ALBUMIN
  , min(pvt2.BICARBONATE) as BICARBONATE
  , min(pvt2.BILIRUBIN) as BILIRUBIN
  , min(pvt2.CREATININE) as CREATININE
  , min(pvt2.CHLORIDE) as CHLORIDE
  , min(pvt2.GLUCOSE) as GLUCOSE
  , min(pvt2.HEMATOCRIT) as HEMATOCRIT
  , min(pvt2.HEMOGLOBIN) as HEMOGLOBIN
  , min(pvt2.LACTATE) as LACTATE
  , min(pvt2.MAGNESIUM) as MAGNESIUM
  , min(pvt2.PHOSPHATE) as PHOSPHATE
  , min(pvt2.PLATELET) as PLATELET
  , min(pvt2.POTASSIUM) as POTASSIUM
  , min(pvt2.PTT) as PTT
  , min(pvt2.INR) as INR
  , min(pvt2.PT) as PT
  , min(pvt2.SODIUM) as SODIUM
  , min(pvt2.BUN) as BUN
  , min(pvt2.WBC) as WBC
-- Easier names
, avg(case when VitalID = 1 then valuenum else null end) as HeartRate_Mean
, avg(case when VitalID = 2 then valuenum else null end) as SysBP_Mean
, avg(case when VitalID = 3 then valuenum else null end) as DiasBP_Mean
, avg(case when VitalID = 4 then valuenum else null end) as MeanBP_Mean
, avg(case when VitalID = 5 then valuenum else null end) as RespRate_Mean
, avg(case when VitalID = 6 then valuenum else null end) as TempC_Mean
, avg(case when VitalID = 7 then valuenum else null end) as SpO2_Mean
, avg(case when VitalID = 8 then valuenum else null end) as Glucose_Mean
FROM  (
  select lb.*
  , round(abs(extract(epoch from lb.event_time - ce.charttime)/3600)) as timepoint
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
  
  from labs lb
  
  LEFT JOIN chartevents ce
  ON lb.subject_id = ce.subject_id 
  AND lb.hadm_id = ce.hadm_id 
  AND lb.icustay_id = ce.icustay_id
  AND ce.charttime between lb.event_time - interval '{0}' hour and lb.event_time - interval '{1}' hour
  -- exclude rows marked as error
  AND ce.error IS DISTINCT FROM 1
  
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
) pvt2
group by subject_id, hadm_id, icustay_id, timepoint
order by subject_id, hadm_id, icustay_id, timepoint;
"""


def vital_and_labs_query(t1, t2):
    """
    SQL QUERY for computing labs stats and hourly vitals stats for mimic patients. 
    Requires 'mv_users' table in database (built using a cohort of patients):
    postgres@/tmp:mimic> \d mimiciii.mv_users;                                                                                                                                                                     
    +--------------+-----------------------------+-------------+
    | Column       | Type                        | Modifiers   |
    |--------------+-----------------------------+-------------|
    | subject_id   | bigint                      |             |
    | hadm_id      | bigint                      |             |
    | icustay_id   | bigint                      |             |
    | ventnum      | double precision            |             |
    | mv_start     | timestamp without time zone |             |
    | transfertime | timestamp without time zone |             |
    +--------------+-----------------------------+-------------+
    
    First argument is WINDOW+LEAD time in hours (e.g. 27).
    Second argument is LEAD time in hours (e.g. 3)
    (http://web.mit.edu/lilehman/www/paper/MV_IEEE_ICHI2018.pdf)
    
    An event time is identified as either the start of mechanical ventilation
    or (for non-ventilated patients)  the 'transfertime' plus WINDOW+LEAD time.
    
    Vitals are computed every hours for all times in the range:
    [event_time - LEAD - WINDOW, event_time-LEAD].
    
    Labs are computed from (if any) max values in the range:
    [beginning of time, event_time-LEAD]
    
    Output dataframe (when running this query) should contain the following columns:
    ['subject_id', 'hadm_id', 'icustay_id', 'timepoint', 'event_time',
     'ventilated', 'mv_start', 'aniongap', 'albumin', 'bicarbonate',
     'bilirubin', 'creatinine', 'chloride', 'glucose', 'hematocrit',
     'hemoglobin', 'lactate', 'magnesium', 'phosphate', 'platelet',
     'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc',
     'heartrate_mean', 'sysbp_mean', 'diasbp_mean', 'meanbp_mean',
      'resprate_mean', 'tempc_mean', 'spo2_mean', 'glucose_mean']   
    """
    return vital_and_labs_template.format(t1, t2)


# FROM https://cs2541-ml4h2020.github.io/#problems
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
      when le.itemid = 50810 and le.valuenum >   100 then null -- percent 'HEMATOCRIT'
      when le.itemid = 51221 and le.valuenum >   100 then null -- percent 'HEMATOCRIT'
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

