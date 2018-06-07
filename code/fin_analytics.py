#Financial analysis

#Assuming a take-rate of 10% and a chargeback loss of 100%

valid_class = valid['Class'].as_data_frame()
valid_amount = valid['Amount'].as_data_frame()
take_rate = 0.1

valid_data = pd.concat([predictions.as_data_frame(), valid_amount, valid_class], axis=1)

total = valid_data.groupby('Class')['Amount'].sum()
print("Fraud: {:06.2f} Gross profit: {:06.2f} Net: {:06.2f}".format(total[1], total[0] * take_rate, (total[0] * take_rate) - total[1]))

def correct_predict(row):
    if row['Class'] == row['predict'] and row['predict'] == 0:
        return row['Amount'] * take_rate
    elif row['Class'] == row['predict'] and row['predict'] == 1:
        return -row['Amount']
    return 0

def missed_profit(row):
    if row['Class'] != row['predict'] and row['predict'] == 0:
        return -row['Amount'] * take_rate
    else:
        return 0

def missed_loss(row):
    if row['Class'] != row['predict'] and row['predict'] == 1:
        return -row['Amount']
    return 0

valid_data['correct_predict'] = valid_data.apply(lambda row: correct_predict(row), axis=1)
valid_data['missed_profit'] = valid_data.apply(lambda row: missed_profit(row), axis=1)
valid_data['missed_loss'] = valid_data.apply(lambda row: missed_loss(row), axis=1)

avoided_loss = valid_data.query('correct_predict < 0')['correct_predict'].sum()
corrected_no_fraud = valid_data.query('correct_predict > 0')['correct_predict'].sum()
missed_profit = valid_data.query('missed_profit < 0')['missed_profit'].sum()
missed_loss = valid_data.query('missed_loss < 0')['missed_loss'].sum()

pd.DataFrame([[-avoided_loss, -missed_profit, -missed_loss, corrected_no_fraud]],
             columns=['avoided loss', 'missed profit', 'missed loss', 'net'])

print("An increase of ${:06.2f} in the net profit".format(754372.79 - 737697.15))
finan
