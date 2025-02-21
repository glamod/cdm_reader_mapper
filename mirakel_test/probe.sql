select
schiffsname as primary_station_id,
tagebuchart,
tagebuchnr,
tagebuchzusatz,
belegbogennr_num,
mess_datum,
mess_datum_moz as report_timestamp,
geogr_breite as longitude,
geogr_laenge as latitude,
lufttemp,
wassertemp,
fahrtrichtung as station_course,
fahrtgeschwindigkeit as station_speed
from
prj_histor.marob_histor mar,
prj_histor.tagebuecher tgb,
schiffe_namen_rufzeichen namen
where
tgb.tagebuch_nummer = mar.tagebuchnr
and tgb.schiffs_id = namen.schiffs_id
and mar.tagebuchart='S'
and tgb.tagebuch_typ_id = 4
and tgb.tagebuch_nummer=5057
;
