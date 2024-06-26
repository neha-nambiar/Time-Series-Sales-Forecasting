# The data is categorized into perishable and non-perishable segments to account for the volatility inherent in perishable products.

import utils

"""
Plots for sales in each state.

"""
for state in states:

    data = sales_items_stores.where(col('state') == state).select('date', 'unit_sales', 'perishable').orderBy('date')
    data = data.toPandas()
    perishable = data[data['perishable']==1].drop('perishable', axis=1)
    not_perishable = data[data['perishable']==0].drop('perishable', axis=1)

    print(state)
    print()

    print('Perishable')
    plot_prediction(not_perishable)

    print('Non-perishable')
    plot_prediction(perishable)
    
    
"""
Plots for sales of each item in each state.

"""
items = distinct_items_df.select('family').collect()
items = [item[0] for item in items]
stores = stores.select('store_id').distinct().collect()
stores = [store_id[0] for store_id in stores]

for store_id in stores:

  filtered_data = sales_items_stores.where(col('store_id') == store_id).select('date', 'unit_sales', 'family') \
    .orderBy('date')
  print(store_id)

  for item in item_list:
    item_data = filtered_data.where(col('family') == item)
    for family in item_data.select('family').distinct().collect():
      family = family[0]
      family_data = item_data.where(col('family') == family).toPandas()
      print(family)
      plot_prediction(family_data)