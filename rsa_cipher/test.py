import rsa
import numpy as np


for _ in range(100):
    (pub, priv) = rsa.newkeys(nbits=256)
    # print(len(pub.save_pkcs1(format="DER")))
    # print(len(list(priv.save_pkcs1(format="DER"))))

    PUB = int.from_bytes(pub.save_pkcs1(format="DER"), "big")
    PRIV =  int.from_bytes(priv.save_pkcs1(format="DER"), "big")

    PUB_VEC = list(pub.save_pkcs1(format="DER"))
    PRIV_VEC = list(priv.save_pkcs1(format="DER"))
    print(len(PUB_VEC), len(PRIV_VEC))
