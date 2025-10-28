class Vehicle:

    def __init__(self, make, name, year, is_electric=False, price=100):
        self.name = name
        self.make = make
        self.year = year
        self.is_electric = is_electric
        self.price = price
        self.odometer = 0

    def drive(self, distance):
        self.odometer += distance

    def compute_price(self):
        if self.odometer == 0:
            price = self.price
        elif self.is_electric:
            price = self.price / (self.odometer * 0.8)
        else:
            price = self.price / self.odometer
        return price


if __name__ == '__main__':
    family_car = Vehicle('Honda', 'Accord', '2019', price=10000)
    family_car_2 = Vehicle(price=100000,
                           name="Honda",
                           year=2020,
                           make='bfgssd')

    print(family_car.compute_price())
    family_car.drive(100)
    print(family_car.compute_price())
