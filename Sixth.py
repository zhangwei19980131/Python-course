has_high_income = True
has_good_credit = True

if has_high_income and has_good_credit:
    print("Eligible for loan")
if has_high_income or has_good_credit:
    print("loan")
# and : both
# or  : at least one
# not
has_criminal_record = False

if has_good_credit and not has_criminal_record:
    print("loan")