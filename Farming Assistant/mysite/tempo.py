 y=x['list'][0]['main']
        temp = y['temp']
        temp_min = y['temp_min']
        temp_max = y['temp_max']
        presssure = y['presssure']
        humidity = y['humidity']
        z=x['list'][0]['weather'][0]
        weather = z['main']
        sky = z['description']
        a=x['list'][0]['weather'][2]
        wind_speed = a['speed']
        degree = a['deg']
        gust = a['gust']
        b = x['list'][0]['weather'][3]
        visibility = b['visibility']
        c = x['list'][0]['weather'][6]
        date = c['dt_txt']
        list = [temp,temp_min,temp_max,presssure,humidity,weather,sky,wind_speed,degree,gust,visibility,date]



'list': [{'dt': 1644051600, 'main': {'temp': 26.82, 'feels_like': 26.73, 'temp_min': 24.49, 'temp_max': 26.82, 'pressure': 1011, 'sea_level': 1011, 'grnd_level': 1009, 'humidity': 40, 'temp_kf': 2.33}, 'weather': [{'id': 802, 'main': 'Clouds', 'description': 'scattered clouds', 'icon': '03d'}], 'clouds': {'all': 44}, 'wind': {'speed': 4.99, 'deg': 292, 'gust': 3.91}, 'visibility': 10000, 'pop': 0, 'sys': {'pod': 'd'}, 'dt_txt': '2022-02-05 09:00:00'},
         {'dt': 1644062400, 'main': {'temp': 25.22, 'feels_like': 25, 'temp_min': 23.84, 'temp_max': 25.22, 'pressure': 1011, 'sea_level': 1011, 'grnd_level': 1009, 'humidity': 46, 'temp_kf': 1.38}, 'weather': [{'id': 803, 'main': 'Clouds', 'description': 'broken clouds', 'icon': '04d'}], 'clouds': {'all': 63}, 'wind': {'speed': 5.69, 'deg': 303, 'gust': 5.57}, 'visibility': 10000, 'pop': 0, 'sys': {'pod': 'd'}, 'dt_txt': '2022-02-05 12:00:00'},
         {'dt': 1644073200, 'main': {'temp': 23, 'feels_like': 22.82, 'temp_min': 23, 'temp_max': 23, 'pressure': 1012, 'sea_level': 1012, 'grnd_level': 1011, 'humidity': 56, 'temp_kf': 0}, 'weather': [{'id': 802, 'main': 'Clouds', 'description': 'scattered clouds', 'icon': '03n'}], 'clouds': {'all': 49}, 'wind': {'speed': 4.88, 'deg': 337, 'gust': 5.12}, 'visibility': 10000, 'pop': 0, 'sys': {'pod': 'n'}, 'dt_txt': '2022-02-05 15:00:00'}