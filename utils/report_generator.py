# utils/report_generator.py
def generate_detailed_report(data):
    # Use NLP to create a narrative report
    report = f"""
    Market Analysis:
    - The population of {data['location']} is {data['population']}, with a median age of {data['median_age']}.

    Economic Indicators:
    - The unemployment rate is {data['unemployment_rate']}%.

    Competitor Analysis:
    - There are {data['competitor_count']} competitors in the area.
    """
    return report
