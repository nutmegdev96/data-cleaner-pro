# Define validation rules
rules = {
    'customer_age': {
        'type': 'in_range',
        'min': 18,
        'max': 100
    },
    'email': {
        'type': 'regex',
        'pattern': r'^[^@]+@[^@]+\.[^@]+$'
    },
    'order_date': {
        'type': 'not_null'
    }
}

# Apply validation
results = cleaner.validate_data(rules)
