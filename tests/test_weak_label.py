from src.weak_label import weak_label

def test_update():
    assert weak_label("My iOS update will not install") == "software_update_issue"

def test_battery():
    assert weak_label("My battery is draining very fast") == "battery_power_charging"

def test_unknown():
    assert weak_label("This is awful") == "general_complaint_unclear"
