import hashlib
import hmac
from crypto_ai_swing.execution.bitvavo import BitvavoREST


def test_signature_matches_documented_concatenation():
    client = BitvavoREST(api_key="k", api_secret="bitvavo")
    ts = 1548172481125
    body = '{"market":"BTC-EUR","side":"buy","price":"5000","amount":"1.23","orderType":"limit"}'
    expected = hmac.new(b"bitvavo", f"{ts}POST/v2/order{body}".encode(), hashlib.sha256).hexdigest()
    assert client.create_signature(ts, "POST", "/order", body) == expected
    client.close()
