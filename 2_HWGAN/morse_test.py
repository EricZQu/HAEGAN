import moses
with open("./sample.txt") as f:
    sms = f.readlines()

metrics = moses.get_all_metrics(sms)

print(metrics)
