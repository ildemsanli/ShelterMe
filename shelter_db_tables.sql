CREATE DATABASE Shelter_db;

USE Shelter_db;

CREATE TABLE Hosts (
host_id INT PRIMARY KEY, 
host_name VARCHAR(100) NOT NULL, 
host_email VARCHAR(100) NOT NULL, 
host_password VARCHAR(20) NOT NULL,
host_phone_number VARCHAR(20) NOT NULL
);

CREATE TABLE Guests (
guest_id INT PRIMARY KEY, 
guest_name VARCHAR(100) NOT NULL, 
guest_email VARCHAR(100) NOT NULL, 
guest_password VARCHAR(20) NOT NULL,
guest_phone_number VARCHAR(20) NOT NULL,
refugee_id VARCHAR(100) NOT NULL, 
arrival_date DATE NOT NULL, 
sex CHAR(5) NOT NULL, 
age INT NOT NULL, 
number_dependants INT NOT NULL, 
number_pets INT NOT NULL
);

CREATE TABLE Booking (
booking_id INT PRIMARY KEY,
isbooked CHAR(5) NOT NULL, 
start_date DATE NOT NULL,
end_date DATE NOT NULL,
guest_id INT, 
FOREIGN KEY (guest_id) REFERENCES Guests(guest_id)
);

CREATE TABLE Accomodation (
room_id INT PRIMARY KEY, 
room_type VARCHAR(20) NOT NULL, 
bathroom VARCHAR(10) NOT NULL, 
kitchen VARCHAR(10) NOT NULL,
address VARCHAR(100) NOT NULL,
postcode VARCHAR(20) NOT NULL, 
city VARCHAR(20) NOT NULL, 
country VARCHAR(20) NOT NULL, 
latitude DECIMAL (10, 8) NOT NULL,
longitude DECIMAL (10, 8) NOT NULL,
host_id INT, 
FOREIGN KEY (host_id) REFERENCES Hosts(host_id),
booking_id INT, 
FOREIGN KEY (booking_id) REFERENCES Booking(booking_id)
);


