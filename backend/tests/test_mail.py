"""Outbound mail: that it is sent, that failure is survivable, and that a
credential stops travelling in the API response once it can travel by email.

No real SMTP server is contacted. `smtplib.SMTP` is replaced with a fake that
records what it was asked to do, which is the only way to assert on STARTTLS and
login ordering — the things that decide whether the password crosses the network
in clear text.
"""
import smtplib

import pytest
from fastapi.testclient import TestClient

from backend import mailer, workspace_routes
from backend.config import settings
from backend.tests.conftest import csrf


class FakeSMTP:
    """Records the calls a send makes. Instances are collected in `sent`."""

    sent: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.started_tls = False
        self.login_args = None
        self.messages = []
        FakeSMTP.sent.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, user, password):
        # Ordering matters: a login before STARTTLS puts the password on the
        # wire in clear text. Recorded so the test can assert it did not happen.
        self.login_args = (user, password, self.started_tls)

    def send_message(self, message):
        self.messages.append(message)


@pytest.fixture
def smtp(monkeypatch):
    """Configure mail and intercept the transport."""
    FakeSMTP.sent = []
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_user", "postmaster@example.com")
    monkeypatch.setattr(settings, "smtp_password", "s3cret-smtp-password")
    monkeypatch.setattr(settings, "mail_from", "Scholar <no-reply@example.com>")
    monkeypatch.setattr(settings, "public_app_url", "https://scholar.example.com")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


# ——— the transport ———


def test_no_provider_configured_is_a_state_not_a_crash(monkeypatch):
    monkeypatch.setattr(settings, "smtp_host", "")
    assert mailer.configured() is False
    assert mailer.send("someone@example.com", "hi", "body") is False


def test_a_from_address_is_required_before_anything_is_sent(monkeypatch):
    """smtp_host alone is not enough — every provider rejects a senderless
    message, so treating mail_from as optional only moves the failure later."""
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "mail_from", "")
    assert mailer.configured() is False


def test_the_password_is_only_sent_after_the_channel_is_encrypted(smtp):
    assert mailer.send("someone@example.com", "Subject", "Body") is True

    (server,) = smtp.sent
    assert server.started_tls, "STARTTLS never ran — the password went out in clear"
    user, password, tls_first = server.login_args
    assert (user, password) == ("postmaster@example.com", "s3cret-smtp-password")
    assert tls_first is True


def test_the_message_carries_the_configured_sender_and_recipient(smtp):
    mailer.send("someone@example.com", "Subject line", "Body text")
    (message,) = smtp.sent[0].messages
    assert message["To"] == "someone@example.com"
    assert message["From"] == "Scholar <no-reply@example.com>"
    assert message["Subject"] == "Subject line"
    assert "Body text" in message.get_content()


def test_a_dead_mail_server_returns_false_rather_than_raising(smtp, monkeypatch):
    def explode(*args, **kwargs):
        raise smtplib.SMTPConnectError(421, "service not available")

    monkeypatch.setattr(smtplib, "SMTP", explode)
    assert mailer.send("someone@example.com", "s", "b") is False


def test_links_use_the_public_url_not_the_cors_origin(smtp, monkeypatch):
    monkeypatch.setattr(settings, "frontend_origin", "http://localhost:5173")
    assert mailer.link("/reset", "abc") == "https://scholar.example.com/reset?token=abc"


def test_links_fall_back_to_the_frontend_origin_when_no_public_url_is_set(monkeypatch):
    monkeypatch.setattr(settings, "public_app_url", "")
    monkeypatch.setattr(settings, "frontend_origin", "http://localhost:5173")
    assert mailer.link("/join", "xyz") == "http://localhost:5173/join?token=xyz"


# ——— password reset ———


def test_a_reset_link_is_emailed_and_actually_works(client: TestClient, smtp):
    client.post("/auth/register", json={"email": "bob@example.com", "password": "validpassword123"})
    client.cookies.clear()

    r = client.post("/auth/forgot", json={"email": "bob@example.com"})
    assert r.status_code == 202

    (message,) = smtp.sent[0].messages
    body = message.get_content()
    assert "https://scholar.example.com/reset?token=" in body

    # The link is the whole point: pull the token out of it exactly as a
    # recipient would, and prove it resets the password.
    token = body.split("/reset?token=")[1].split()[0]
    r = client.post("/auth/reset", json={"token": token, "password": "brandnewpassword"})
    assert r.status_code == 200, r.text

    client.cookies.clear()
    assert client.post(
        "/auth/login", json={"email": "bob@example.com", "password": "brandnewpassword"}
    ).status_code == 200


def test_an_unknown_address_sends_nothing_but_answers_the_same(client: TestClient, smtp):
    """The 202 is uniform so the route is not an account-existence oracle. That
    only holds if a missing account also produces no mail-shaped delay or error."""
    r = client.post("/auth/forgot", json={"email": "nobody@example.com"})
    assert r.status_code == 202
    assert smtp.sent == []


def test_a_failed_send_does_not_fall_back_to_logging_the_token(
    client: TestClient, smtp, monkeypatch, caplog
):
    """Configuring mail is the operator saying tokens must not reach the logs.
    A transient SMTP outage does not revoke that instruction."""
    client.post("/auth/register", json={"email": "bob@example.com", "password": "validpassword123"})
    monkeypatch.setattr(mailer, "send", lambda *a, **k: False)

    with caplog.at_level("WARNING"):
        assert client.post("/auth/forgot", json={"email": "bob@example.com"}).status_code == 202

    assert "token logged instead" not in caplog.text


def test_without_a_provider_the_token_is_logged_and_says_so(
    client: TestClient, monkeypatch, caplog
):
    monkeypatch.setattr(settings, "smtp_host", "")
    client.post("/auth/register", json={"email": "bob@example.com", "password": "validpassword123"})

    with caplog.at_level("WARNING"):
        client.post("/auth/forgot", json={"email": "bob@example.com"})

    assert "no mail provider configured" in caplog.text


# ——— invitations ———


def _team(alice: TestClient) -> int:
    return alice.post("/workspaces", json={"name": "Acme Legal"}, headers=csrf(alice)).json()["id"]


def test_an_emailed_invitation_keeps_its_token_out_of_the_response(alice: TestClient, smtp):
    workspace_id = _team(alice)
    r = alice.post(f"/workspaces/{workspace_id}/invitations",
                   json={"email": "carol@example.com"}, headers=csrf(alice))
    assert r.status_code == 201, r.text

    body = r.json()
    assert body["delivered"] is True
    assert body["token"] is None, "an invitation token is a credential; it left by email"

    (message,) = smtp.sent[0].messages
    assert message["To"] == "carol@example.com"
    assert "https://scholar.example.com/join?token=" in message.get_content()


def test_the_emailed_invitation_token_is_the_one_that_works(
    alice: TestClient, other_client: TestClient, smtp
):
    workspace_id = _team(alice)
    alice.post(f"/workspaces/{workspace_id}/invitations",
               json={"email": "other@example.com"}, headers=csrf(alice))
    token = smtp.sent[0].messages[0].get_content().split("/join?token=")[1].split()[0]

    other_client.post("/auth/register",
                      json={"email": "other@example.com", "password": "validpassword123"})
    r = other_client.post("/workspaces/accept", json={"token": token}, headers=csrf(other_client))
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Acme Legal"


def test_without_a_provider_the_inviter_still_gets_the_code(alice: TestClient, monkeypatch):
    """The stopgap survives only while it is the ONLY delivery route there is."""
    monkeypatch.setattr(settings, "smtp_host", "")
    workspace_id = _team(alice)
    body = alice.post(f"/workspaces/{workspace_id}/invitations",
                      json={"email": "carol@example.com"}, headers=csrf(alice)).json()
    assert body["delivered"] is False
    assert body["token"]


def test_a_failed_invitation_send_hands_the_code_back(alice: TestClient, smtp, monkeypatch):
    """Unlike a password reset, an invitation has a safe human fallback: the
    inviter already knows who they invited, and can pass the code on directly."""
    monkeypatch.setattr(mailer, "send", lambda *a, **k: False)
    workspace_id = _team(alice)
    body = alice.post(f"/workspaces/{workspace_id}/invitations",
                      json={"email": "carol@example.com"}, headers=csrf(alice)).json()
    assert body["delivered"] is False
    assert body["token"]


def test_the_invitation_names_the_inviter_and_the_workspace(alice: TestClient, smtp):
    workspace_id = _team(alice)
    alice.post(f"/workspaces/{workspace_id}/invitations",
               json={"email": "carol@example.com"}, headers=csrf(alice))

    message = smtp.sent[0].messages[0]
    assert "Acme Legal" in message["Subject"]
    body = message.get_content()
    assert "alice@example.com" in body, "an invitation from nobody is indistinguishable from spam"


def test_delivery_reports_honestly_when_it_did_nothing(monkeypatch):
    """`deliver_invitation` returning True is what removes the token from the
    API response. If it ever lied, the invitee would receive nothing and have
    no way to be sent it."""
    monkeypatch.setattr(settings, "smtp_host", "")

    class W:
        name = "Acme"

    class U:
        email = "alice@example.com"

    assert workspace_routes.deliver_invitation("c@example.com", W(), U(), "tok") is False
