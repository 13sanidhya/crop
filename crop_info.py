# Crop Information Database
# Contains detailed information about each crop

CROP_INFO = {
    1: {
        "name": "Rice",
        "season": "Kharif (June-October)",
        "water_needs": "High (1200-1800mm)",
        "soil_type": "Clayey loam",
        "growing_period": "3-6 months",
        "temperature": "20-35°C",
        "description": "Rice is a staple food crop requiring standing water and warm climate.",
        "fertilizer": "NPK 120:60:40 kg/ha",
        "market_price": "₹2000-2500 per quintal"
    },
    2: {
        "name": "Maize",
        "season": "Kharif & Rabi",
        "water_needs": "Medium (500-800mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "3-4 months",
        "temperature": "21-27°C",
        "description": "Maize is a versatile cereal crop used for food, feed, and industrial purposes.",
        "fertilizer": "NPK 120:60:40 kg/ha",
        "market_price": "₹1800-2200 per quintal"
    },
    3: {
        "name": "Jute",
        "season": "Kharif (March-July)",
        "water_needs": "High (1500-2000mm)",
        "soil_type": "Alluvial soil",
        "growing_period": "4-5 months",
        "temperature": "24-37°C",
        "description": "Jute is a fiber crop requiring warm and humid climate with adequate rainfall.",
        "fertilizer": "NPK 60:30:30 kg/ha",
        "market_price": "₹4000-5000 per quintal"
    },
    4: {
        "name": "Cotton",
        "season": "Kharif (April-October)",
        "water_needs": "Medium (600-1200mm)",
        "soil_type": "Deep black soil",
        "growing_period": "5-6 months",
        "temperature": "21-30°C",
        "description": "Cotton is a major fiber crop requiring long frost-free period.",
        "fertilizer": "NPK 120:60:60 kg/ha",
        "market_price": "₹5500-6500 per quintal"
    },
    5: {
        "name": "Coconut",
        "season": "Year-round",
        "water_needs": "High (1500-2500mm)",
        "soil_type": "Sandy loam coastal",
        "growing_period": "Perennial",
        "temperature": "27-32°C",
        "description": "Coconut is a tropical palm crop thriving in coastal areas with high humidity.",
        "fertilizer": "NPK 600:300:1200 g/palm/year",
        "market_price": "₹15-35 per coconut"
    },
    6: {
        "name": "Papaya",
        "season": "Year-round",
        "water_needs": "Medium (1000-1500mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "9-12 months",
        "temperature": "25-30°C",
        "description": "Papaya is a fast-growing fruit crop suitable for tropical and subtropical regions.",
        "fertilizer": "NPK 200:200:400 g/plant",
        "market_price": "₹15-25 per kg"
    },
    7: {
        "name": "Orange",
        "season": "Winter harvest",
        "water_needs": "Medium (900-1200mm)",
        "soil_type": "Well-drained sandy loam",
        "growing_period": "Perennial",
        "temperature": "15-30°C",
        "description": "Orange is a citrus fruit crop requiring well-distributed rainfall and cool winters.",
        "fertilizer": "NPK 500:250:500 g/tree/year",
        "market_price": "₹30-50 per kg"
    },
    8: {
        "name": "Apple",
        "season": "Autumn harvest",
        "water_needs": "Medium (1000-1250mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "Perennial",
        "temperature": "15-24°C",
        "description": "Apple requires cold winters for dormancy and moderate summers for fruit development.",
        "fertilizer": "NPK 450:225:450 g/tree/year",
        "market_price": "₹60-100 per kg"
    },
    9: {
        "name": "Muskmelon",
        "season": "Summer (Feb-June)",
        "water_needs": "Medium (400-600mm)",
        "soil_type": "Sandy loam",
        "growing_period": "3-4 months",
        "temperature": "25-35°C",
        "description": "Muskmelon is a warm-season crop requiring dry weather during fruit maturity.",
        "fertilizer": "NPK 100:50:75 kg/ha",
        "market_price": "₹20-40 per kg"
    },
    10: {
        "name": "Watermelon",
        "season": "Summer (Feb-June)",
        "water_needs": "Medium (450-600mm)",
        "soil_type": "Sandy loam",
        "growing_period": "3-4 months",
        "temperature": "24-30°C",
        "description": "Watermelon thrives in warm weather with long sunny days and moderate water.",
        "fertilizer": "NPK 100:50:75 kg/ha",
        "market_price": "₹10-20 per kg"
    },
    11: {
        "name": "Grapes",
        "season": "Year-round (varies)",
        "water_needs": "Medium (600-900mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "Perennial",
        "temperature": "15-35°C",
        "description": "Grapes require dry summers and cool winters with good sunlight exposure.",
        "fertilizer": "NPK 400:200:400 g/vine/year",
        "market_price": "₹40-80 per kg"
    },
    12: {
        "name": "Mango",
        "season": "Summer (April-July)",
        "water_needs": "Medium (750-1500mm)",
        "soil_type": "Well-drained deep loamy",
        "growing_period": "Perennial",
        "temperature": "24-30°C",
        "description": "Mango is the king of fruits, requiring tropical to subtropical climate.",
        "fertilizer": "NPK 1000:500:1000 g/tree/year",
        "market_price": "₹40-100 per kg"
    },
    13: {
        "name": "Banana",
        "season": "Year-round",
        "water_needs": "High (1500-3000mm)",
        "soil_type": "Rich loamy with good drainage",
        "growing_period": "9-12 months",
        "temperature": "20-35°C",
        "description": "Banana is a fast-growing tropical fruit crop requiring high moisture.",
        "fertilizer": "NPK 200:60:200 g/plant",
        "market_price": "₹20-40 per dozen"
    },
    14: {
        "name": "Pomegranate",
        "season": "Twice a year",
        "water_needs": "Low-Medium (500-700mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "Perennial",
        "temperature": "15-38°C",
        "description": "Pomegranate is drought-tolerant and thrives in semi-arid conditions.",
        "fertilizer": "NPK 500:250:500 g/plant/year",
        "market_price": "₹60-120 per kg"
    },
    15: {
        "name": "Lentil",
        "season": "Rabi (Oct-April)",
        "water_needs": "Low (300-400mm)",
        "soil_type": "Loamy soil",
        "growing_period": "4-5 months",
        "temperature": "18-25°C",
        "description": "Lentil is a cool-season pulse crop requiring minimal irrigation.",
        "fertilizer": "NPK 20:40:20 kg/ha",
        "market_price": "₹5000-7000 per quintal"
    },
    16: {
        "name": "Blackgram",
        "season": "Kharif & Rabi",
        "water_needs": "Medium (600-900mm)",
        "soil_type": "Loamy soil",
        "growing_period": "2.5-3 months",
        "temperature": "25-35°C",
        "description": "Blackgram is a short-duration pulse crop with nitrogen-fixing ability.",
        "fertilizer": "NPK 20:40:20 kg/ha",
        "market_price": "₹6000-8000 per quintal"
    },
    17: {
        "name": "Mungbean",
        "season": "Kharif & Summer",
        "water_needs": "Medium (600-900mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "2-3 months",
        "temperature": "25-35°C",
        "description": "Mungbean is a short-duration pulse crop suitable for multiple cropping.",
        "fertilizer": "NPK 20:40:20 kg/ha",
        "market_price": "₹6500-8500 per quintal"
    },
    18: {
        "name": "Mothbeans",
        "season": "Kharif (July-Oct)",
        "water_needs": "Low (300-500mm)",
        "soil_type": "Sandy loam",
        "growing_period": "3-4 months",
        "temperature": "25-35°C",
        "description": "Mothbeans are drought-resistant pulses suitable for arid regions.",
        "fertilizer": "NPK 15:30:15 kg/ha",
        "market_price": "₹4500-6000 per quintal"
    },
    19: {
        "name": "Pigeonpeas",
        "season": "Kharif (June-March)",
        "water_needs": "Medium (600-1000mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "5-7 months",
        "temperature": "20-35°C",
        "description": "Pigeonpeas are long-duration pulse crops with deep root system.",
        "fertilizer": "NPK 20:50:20 kg/ha",
        "market_price": "₹5500-7500 per quintal"
    },
    20: {
        "name": "Kidneybeans",
        "season": "Rabi (Oct-March)",
        "water_needs": "Medium (500-700mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "3-4 months",
        "temperature": "15-25°C",
        "description": "Kidneybeans are cool-season pulses with high protein content.",
        "fertilizer": "NPK 20:40:20 kg/ha",
        "market_price": "₹7000-9000 per quintal"
    },
    21: {
        "name": "Chickpea",
        "season": "Rabi (Oct-April)",
        "water_needs": "Low-Medium (300-500mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "4-5 months",
        "temperature": "20-30°C",
        "description": "Chickpea is a major pulse crop suitable for dryland farming.",
        "fertilizer": "NPK 20:40:20 kg/ha",
        "market_price": "₹5000-6500 per quintal"
    },
    22: {
        "name": "Coffee",
        "season": "Year-round",
        "water_needs": "High (1500-2000mm)",
        "soil_type": "Well-drained loamy",
        "growing_period": "Perennial",
        "temperature": "15-28°C",
        "description": "Coffee requires shade, high altitude, and well-distributed rainfall.",
        "fertilizer": "NPK 100:50:100 g/plant/year",
        "market_price": "₹15000-25000 per quintal"
    }
}
