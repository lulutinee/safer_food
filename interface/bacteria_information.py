"""
Thresholds for the different microorganims
Structure : dict {'<bacteria code>': {
    'usual_name': '<usual name of the microorganism>,
    'quality_threshold': <concentration in log cfu/g or /ml inducing quality change,
    'danger_threshold': <infectious dose in log cfu/g or /ml for an healthy adult>
    }}

Data on infectious dose are from Health Canada pathogen safety data and MAPAQ (for total flora)
Amount of food ingested is considered to be 1 kg or 1 L (1e3 g), Infectious dose is calculated
on this amount of food
Initial bacterial load for pathogens is considered to be "absent in 25g" = 39 for 1e3 g of food = -1.4 log
"""

#Pathogens infectious dose Source : Health Canada (HC)
# Listeria : 10e6 to 100e6 /g https://www.canada.ca/en/public-health/services/laboratory-biosafety-biosecurity/pathogen-safety-data-sheets-risk-assessment/listeria-monocytogenes.html
# Salmonella : 10e3 to 1e5 https://www.canada.ca/en/public-health/services/laboratory-biosafety-biosecurity/pathogen-safety-data-sheets-risk-assessment/salmonella-enterica.html
# E. coli : 10 https://www.canada.ca/en/public-health/services/laboratory-biosafety-biosecurity/pathogen-safety-data-sheets-risk-assessment/escherichia-coli-enterohemorrhagic.html

#Food matrices quality parameters : MAPAQ CUMAIRA https://numerique.banq.qc.ca/patrimoine/details/52327/4517890


MICROORGANISM = {
    'lm': {
        'usual_name': 'Listeria monocytogenes',
        'initial': -1.4,
        'raw': 4,
        'medium': 4,
        'fried': 4
    },
    'ss': {
        'usual_name': 'Salmonella',
        'initial': -1.4,
        'raw': 1,
        'medium': 1,
        'fried': 1
    },
    'ec': {
        'usual_name': 'Escherichia coli',
        'initial': -1.4,
        'raw': -2,
        'medium': -2,
        'fried': -2
    },
    'ta': {
        'usual_name': 'Total bacteria (aerobic mesophilic)',
        'initial': 4,
        'raw': 7,
        'medium': 8,
        'fried': 9
    }
}
