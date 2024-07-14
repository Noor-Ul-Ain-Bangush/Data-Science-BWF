-- Create the 'cities' table
CREATE TABLE cities (
    city_id INT PRIMARY KEY,
    city VARCHAR(50),
    country VARCHAR(50)
);

-- Insert data into the 'cities' table
INSERT INTO cities (city_id, city, country) VALUES
(1, 'Tokyo', 'Japan'),
(2, 'Atlanta', 'United States'),
(3, 'Auckland', 'New Zealand');

-- Create the 'rainfall' table
CREATE TABLE rainfall (
    rainfall_id INT PRIMARY KEY,
    city_id INT,
    year INT,
    amount INT,
    FOREIGN KEY (city_id) REFERENCES cities(city_id)
);

-- Insert data into the 'rainfall' table
INSERT INTO rainfall (rainfall_id, city_id, year, amount) VALUES
(1, 1, 2018, 1445),
(2, 1, 2019, 1874),
(3, 1, 2020, 1690),
(4, 2, 2018, 1779),
(5, 2, 2019, 1111),
(6, 2, 2020, 1683),
(7, 3, 2018, 1386),
(8, 3, 2019, 942),
(9, 3, 2020, 1176);

-- Retrieve all cities
SELECT city
FROM cities;

-- Retrieve cities in New Zealand
SELECT city
FROM cities
WHERE country = 'New Zealand';

-- Retrieve the rainfall for 2019 for all cities
SELECT cities.city, rainfall.amount
FROM cities
INNER JOIN rainfall ON cities.city_id = rainfall.city_id
WHERE rainfall.year = 2019;
