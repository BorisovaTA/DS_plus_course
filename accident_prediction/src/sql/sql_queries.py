GET_ALL_TABLES = '''
SELECT table_name
FROM information_schema.tables
WHERE table_schema='public'
ORDER BY table_name;
'''

GET_CASE_IDS = '''
SELECT * 
FROM case_ids
LIMIT 5;
'''

GET_VEHICLES = '''
SELECT * 
FROM vehicles
LIMIT 5;
'''

GET_COLLISIONS = '''
SELECT * 
FROM collisions
LIMIT 5;
''' 

GET_PARTIES = '''
SELECT * 
FROM parties
LIMIT 5;
''' 

GET_TABLES_SHAPE = '''
SELECT 'collisions' AS table, COUNT(*) AS rows, COUNT(DISTINCT case_id) AS uniq_case
FROM collisions
UNION ALL
SELECT 'parties', COUNT(*), COUNT(DISTINCT case_id)
FROM parties
UNION ALL
SELECT 'vehicles', COUNT(*), COUNT(DISTINCT case_id)
FROM vehicles
UNION ALL
SELECT 'case_ids', COUNT(*), COUNT(DISTINCT case_id)
FROM case_ids
''' 

GET_DATES = '''
SELECT MIN(collision_date) AS min_date, MAX(collision_date) AS max_date
FROM collisions;
''' 

GET_COLLISIONS_PER_MONTH = """
SELECT DISTINCT EXTRACT(MONTH FROM collision_date)::int AS month,
       COUNT(case_id) OVER(PARTITION BY EXTRACT(MONTH FROM collision_date)) AS qty_collisions
FROM collisions
ORDER BY month;
"""

GET_AGE_FAULT = '''
WITH car_info AS (SELECT p.case_id,
                         p.party_number,
                         p.party_type,
                         p.at_fault,
                         v.vehicle_age
                  FROM parties p
                  JOIN vehicles v ON p.case_id=v.case_id AND p.party_number=v.party_number
                  WHERE p.party_type ='car' AND v.vehicle_age IS NOT NULL
                  ),
     faults_info AS (SELECT DISTINCT vehicle_age, 
                            COUNT(case_id) OVER(PARTITION BY vehicle_age) AS qty_collisions,
                            SUM(at_fault) OVER(PARTITION BY vehicle_age) AS qty_faults
                     FROM car_info)
                  
SELECT *,
       ROUND(100.0 * qty_faults/qty_collisions, 2) AS fault_rate
FROM faults_info
ORDER BY vehicle_age
'''

GET_TRANSMISSION_FAULT = '''
WITH car_info AS (SELECT p.case_id,
                         p.party_number,
                         p.party_type,
                         p.at_fault,
                         v.vehicle_transmission
                  FROM parties p
                  JOIN vehicles v ON p.case_id=v.case_id AND p.party_number=v.party_number
                  WHERE p.party_type ='car' AND v.vehicle_transmission IS NOT NULL
                  ),
     faults_info AS (SELECT DISTINCT vehicle_transmission, 
                            COUNT(case_id) OVER(PARTITION BY vehicle_transmission) AS qty_collisions,
                            SUM(at_fault) OVER(PARTITION BY vehicle_transmission) AS qty_faults
                     FROM car_info)
                  
SELECT *,
       ROUND(100.0 * qty_faults/qty_collisions, 2) AS fault_rate
FROM faults_info;
'''

GET_FINAL_DATA = '''
SELECT     
    -- ТАБЛИЦА COLLISIONS ---
    c.weather_1,             -- Погода во время ДТП
    c.road_surface,          -- Тип покрытия
    c.road_condition_1,      -- Состояние дороги
    c.lighting,              -- Освещенность
    c.distance,              -- Расстояние от главной дороги, м
    
    -- ТАБЛИЦА PARTIES ---
    p.cellphone_in_use,      -- Наличие телефона с громкой связью
    p.at_fault,              -- Целевая переменная
    
    -- ТАБЛИЦА VEHICLES ---
    v.vehicle_type,          -- Тип кузова
    v.vehicle_transmission,  -- Тип коробки
    v.vehicle_age            -- Возраст авто

FROM collisions c
JOIN parties p
    ON c.case_id = p.case_id
JOIN vehicles v
    ON c.case_id = v.case_id AND p.party_number = v.party_number
WHERE p.party_type = 'car'
  AND c.collision_damage != 'scratch'
  AND EXTRACT(YEAR FROM collision_date::date) = 2012;
'''
