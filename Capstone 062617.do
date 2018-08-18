use "C:\Users\david_000\Google Drive\HSLS_2009_v3_0_Stata_Datasets\hsls_09_student_v3_0.dta", clear
keep if X1POVERTY185 == 1


foreach var of varlist X1TXMTH {
  replace `var' = . if `var' == -8
  summ `var'
}



foreach var of varlist S1MCLUB S1MCOMPETE S1MCAMP S1MTUTOR S1M8 S1M8GRADE S1MFALL09 S1ALG1M09 S1GEOM09 S1ALG2M09 S1TRIGM09 S1REVM09 S1INTGM109 S1STATSM09 S1INTGM209 S1PREALGM09 S1ANGEOM09 S1ADVM09 S1OTHM09  {
  replace `var' = . if `var' < 0
  tab `var'
}

foreach var of varlist X3TGPAENG X3T1CREDALG1 X3T1CREDALG2 X3T1CREDINTM X3T1CREDPREC X3TCREDAPMTH X3T1CREDCALC X3T1CREDGEO X3T1CREDSTAT X3T1CREDTRIG X3TCREDMAT X3THIMATH X3THIMATH9 X3TGPAMAT X3TGPAHIMTH X3TWHENALG1 {
  replace `var' = . if `var' < 0
  tab `var'
}

tab S3CLASSES S3FIELD_STEM, nol

drop if S3FIELD_STEM < 0 & S3FIELD_STEM != -7

foreach var of varlist S3CLASSES S3CLGFT S3FIELD_STEM {
  replace `var' = . if `var' < 0
  tab `var'

}

