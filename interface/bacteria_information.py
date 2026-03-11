"""
Thresholds for the different microorganims
Structure : dict {
        '<bacteria code>': {                # bacteria code (ss = salmonella, ec : E. coli, lm : Listeria monocytogenes, etc.)
        'usual_name': 'Escherichia coli',   # usual name of the bacteria
        'initial': -1.4,                    # expected initial load on commercial foods (log cfu/g /ml)
        'raw': -2,                          # maximum load acceptable for raw consumption
        'medium': -2,                       # maximum load acceptable for products if they are cooked at medium temperature
        'fried': -2                         # maximum load acceptable for products if they are coocked at high temperature
    }

For pathogens : Data on infectious dose are from Health Canada pathogen safety data
For spoilage (total aerobic mesophilic) : MAPAQ (CUMAIRA)
Amount of food ingested is considered to be 1 kg or 1 L (1e3 g), Infectious dose is calculated
on this amount of food
Initial bacterial load for pathogens is considered to be "absent in 25g" = 39 cfu for 1e3 g of food = -1.4 log for Listeria and Salmonella, "1 in 1 kg" for E. coli
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
        'raw': 1,
        'medium': 3,
        'high': 4
    },
    'ss': {
        'usual_name': 'Salmonella enterica',
        'initial': -1.4,
        'raw': 0.2,
        'medium': 0.7,
        'high': 1
    },
    'ec': {
        'usual_name': 'Escherichia coli',
        'initial': -3,
        'raw': -2.7,
        'medium': -2.3,
        'high': -2
    },
    'ta': {
        'usual_name': 'Total bacteria (aerobic mesophilic)',
        'initial': 4,
        'raw': 6,
        'medium': 8,
        'high': 9
    }
}
