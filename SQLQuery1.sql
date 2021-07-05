use UC
select * from Tesla
order by Date desc ;
;
SELECT COLUMN_NAME as cols
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'Tesla'
;


GO;

CREATE PROCEDURE five_point @Open_Price nvarchar(50)
AS
go
;
SELECT min(High_Price) as High_min, max(High_Price) as High_max,avg(High_Price) as High_avg FROM Tesla
SELECT min(Low_Price) as Low_min, max(Low_Price) as Low_max,avg(Low_Price) as Low_avg FROM Tesla
SELECT min(Open_Price) as Open_min, max(Open_Price) as Open_max,avg(Open_Price) as Open_avg FROM Tesla
SELECT min(Close_Price) as Close_min, max(Close_Price) as Close_max,avg(Close_Price) as Close_avg FROM Tesla
SELECT min(Volume) as Vol_min, max(Volume) as Vol_max,avg(Volume) as Vol_avg FROM Tesla
SELECT min(daily_percent_change) as chg_min, max(daily_percent_change) as chg_max,avg(daily_percent_change) as chg_avg FROM Tesla

;

alter table Tesla
add constraint id_null check(F1 is not null) 
alter table Tesla
drop constraint id_null
alter table Tesla
alter column F1 int NOT NULL
alter table Tesla
add primary key(F1)
;
/* test insertion */

insert into Tesla
values(2676,'2021-02-12 00:00:00.000',805,825,800,801,802,400000,.8,845)
/* test deletion */
delete from Tesla
where F1=2676
/* Finding min and max high prices */
select max(High_Price) as MaxHigh
from Tesla
select min((High_Price)) as MinHigh
from Tesla
where year(Date)<2019
/* Sorting table by High Price */
select * from Tesla
order by High_Price ;
;
/*finding maximum high prices and volumes by year */

select year(Date),max(High_Price) as MaxHighPrice,max(Volume) as MaxVolume
from Tesla
group by year(Date)
order by max(Volume) desc
/* cast/convert */

select Date,F1,convert(int,Open_Price) as RoundedIntegerHighPrice
from Tesla

/* Case expressions to find rising valuation */
alter table Tesla
add ValuationClass varchar(100)
alter table Tesla
add constraint val
check (ValuationClass in ('Low','Medium','High'))

/* new column to categorise valuation */
update Tesla
set ValuationClass= case 
when cum_percent_change<=100
then 'Low'
when cum_percent_change<=400
then 'Medium'
else 'High'
end
