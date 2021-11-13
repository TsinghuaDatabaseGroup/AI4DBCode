from datetime import date, timedelta
from random import randint


def rand_shipdate_in(year):
    s = date(year, 1, 1) + timedelta(randint(0, 364 - 30))
    e = s + timedelta(30)
    return str(s), str(e)


def rand_discount():
    s = float(randint(0, 8)) * 0.01
    return str(s), str(s + 0.02)


shipmode = ['TRUCK', 'AIR', 'FOB', 'RAIL', 'SHIP', 'MAIL', 'REGAIR']


def rand_shipmode():
    return shipmode[randint(0, len(shipmode) - 1)]


def gen_snippet(year):
    ys, ye = rand_shipdate_in(1992)
    ds, de = rand_discount()
    sm = rand_shipmode()
    return (ys, ye, ds, de, sm)


for i in range(10):
    (ys, ye, ds, de) = gen_snippet(1992)

    q = '''select l_shipmode, sum(h_count) from hist
    where l_shipdate between '%s' and '%s' and
          l_discount between %s and %s
    group by l_shipmode
    order by l_shipmode;''' % (ys, ye, ds, de)
    print
    q
    print
