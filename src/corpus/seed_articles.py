"""StreamFlix help-center corpus — seed content.

45 articles across 6 categories, written to mirror a real support
knowledge base including the flaws real ones have.

DELIBERATE IMPERFECTIONS (these are the point, not bugs):

  1. CONTRADICTION — `bill-002` states a 30-day refund window;
     `bill-003` states 14 days. Real help centers contradict themselves
     when policies change and not every page gets updated. Tests whether
     the system notices conflict or confidently picks one at random.

  2. OUTDATED — `bill-009` documents the "Basic Plus" tier, discontinued
     in 2025. Still live in the corpus, as stale pages are in reality.
     Tests whether retrieval surfaces obsolete information.

  3. NEAR-DUPLICATES — `dev-003`/`dev-004`/`dev-005` (Roku / Apple TV /
     Fire TV) are structurally near-identical. Tests retrieval precision:
     does a Roku question surface the Roku article, or a sibling?

  4. PARTIAL COVERAGE — several topics are mentioned in passing but not
     fully documented, forcing multi-hop retrieval.

  5. COVERAGE GAPS — no article covers gift subscriptions, business
     accounts, accessibility features, live sports, password-sharing
     enforcement, or GDPR data export. These generate the out-of-scope
     questions that test refusal behaviour.

Fields:
    article_id    stable identifier, used as retrieval ground truth
    title         human-readable heading
    category      one of the six top-level sections
    last_updated  ISO date; deliberately stale for some articles
    body          markdown content
"""
from __future__ import annotations

ARTICLES: list[dict] = [

    # =================================================================
    # BILLING & PAYMENTS
    # =================================================================
    {
        "article_id": "bill-001",
        "title": "How StreamFlix billing works",
        "category": "billing",
        "last_updated": "2026-03-14",
        "body": """
StreamFlix bills on a monthly recurring cycle. Your billing date is set
by the day you first subscribed and stays the same each month.

**Billing cycle basics**

- Charges post on the same calendar day each month. If your billing day
  is the 31st, months with fewer days bill on the final day of the month.
- Payment is taken automatically from the method on file.
- Your plan is charged in advance — the charge on 1 April covers service
  through 30 April.

**Plan pricing**

| Plan | Monthly price | Simultaneous streams |
|---|---|---|
| Basic | $9 | 1 |
| Standard | $14 | 2 |
| Premium | $19 | 4 |

Prices shown exclude tax. See "Taxes and regional pricing" for how tax
is applied in your region.

**Where to find your charges**

Open Account → Billing → Payment history. Each entry shows the amount,
date, plan, and the last four digits of the card used.
""",
    },
    {
        "article_id": "bill-002",
        "title": "Refund policy",
        "category": "billing",
        "last_updated": "2025-08-02",
        "body": """
StreamFlix offers refunds on subscription charges in limited
circumstances.

**Standard refund window**

You may request a refund within **30 days** of a charge if you have not
streamed any content during the billing period covered by that charge.
Requests are reviewed by our billing team and typically processed within
5–7 business days.

**Eligible situations**

- Duplicate charge for the same billing period
- Charge after a confirmed cancellation
- Service outage exceeding 24 continuous hours
- Unauthorised charge on a compromised account

**Not eligible**

- Change of mind after streaming content during the period
- Partial-month refunds for mid-cycle cancellation
- Promotional periods that have already been consumed

**How to request**

Contact support through Account → Help → Contact us and include the
charge date and amount. Refunds are returned to the original payment
method.
""",
    },
    {
        "article_id": "bill-003",
        "title": "Canceling your subscription",
        "category": "billing",
        "last_updated": "2026-01-20",
        "body": """
You can cancel your StreamFlix subscription at any time. There is no
cancellation fee and no minimum contract.

**How to cancel**

1. Open Account → Membership
2. Select **Cancel membership**
3. Confirm the cancellation

You will receive a confirmation email. If you do not receive one within
an hour, the cancellation did not complete.

**What happens next**

Your subscription remains active until the end of the current billing
period. You keep full access until that date — cancelling on the 3rd
when your cycle ends on the 28th means you retain access through the
28th.

**Refunds after cancellation**

Cancellation does not automatically trigger a refund. If you were
charged in error, you may request a refund within **14 days** of the
charge date. Refund requests outside this window are not accepted.

**Restarting**

Your viewing history and profiles are retained for 10 months after
cancellation. Resubscribing within that window restores them.
""",
    },
    {
        "article_id": "bill-004",
        "title": "Failed or declined payments",
        "category": "billing",
        "last_updated": "2026-02-11",
        "body": """
If a payment fails, StreamFlix will retry the charge and notify you by
email.

**Retry schedule**

We attempt the charge up to four times over eight days: on the billing
date, then at days 3, 5, and 8. Your account remains active during this
window.

**If all retries fail**

Access is suspended on day 9. Your profiles, viewing history, and My
List are preserved. Updating a valid payment method restores access
immediately and the outstanding charge is taken at that point.

**Common causes**

- Card expired or was reissued with a new number
- Insufficient funds at the time of the attempt
- Bank declined the transaction as suspected fraud
- Billing address on file no longer matches the card

**Resolving**

Go to Account → Billing → Payment method and add a current card. If the
card is valid and the charge still fails, contact your bank — some
issuers block recurring international merchants by default.
""",
    },
    {
        "article_id": "bill-005",
        "title": "Updating your payment method",
        "category": "billing",
        "last_updated": "2026-02-11",
        "body": """
You can change the card or payment method on your account at any time.

**Steps**

1. Account → Billing → Payment method
2. Select **Add payment method**
3. Enter the card details and billing address
4. Set it as the default method

The new method is used from your next billing date. If a payment is
currently in a failed-retry state, adding a valid method charges it
immediately and restores access.

**Accepted payment methods**

- Visa, Mastercard, American Express
- Debit cards carrying a Visa or Mastercard logo
- PayPal (in supported regions)

We do not accept prepaid cards, virtual card numbers that expire before
the next billing date, or direct bank transfer.

**Billing address**

The address must match what your card issuer holds. A mismatch is one
of the most common causes of an otherwise-valid card being declined.
""",
    },
    {
        "article_id": "bill-006",
        "title": "Understanding your invoice",
        "category": "billing",
        "last_updated": "2025-11-30",
        "body": """
Each charge generates an invoice available in Account → Billing →
Payment history.

**What an invoice shows**

- Billing period covered (start and end date)
- Plan name and monthly rate
- Any promotional discount applied, shown as a separate line
- Tax, itemised by rate and jurisdiction
- Total charged and payment method last four digits

**Why the amount may differ month to month**

- A promotional rate ended
- You changed plan tier mid-cycle, producing a prorated line
- Local tax rate changed
- Currency conversion, if your card is issued outside the billing region

**Downloading invoices**

Select any entry in Payment history and choose **Download PDF**. Invoices
are retained for 24 months.
""",
    },
    {
        "article_id": "bill-007",
        "title": "Changing your plan",
        "category": "billing",
        "last_updated": "2026-01-08",
        "body": """
You can move between Basic, Standard, and Premium at any time.

**Upgrading**

Upgrades take effect immediately. You are charged a prorated amount for
the remainder of the current cycle, and the new rate applies from the
next billing date. Extra simultaneous streams and higher resolution
become available right away.

**Downgrading**

Downgrades take effect at the end of the current billing period, so you
keep the higher tier for the time already paid for. No proration or
partial refund is issued.

**How to change**

Account → Membership → Change plan, then select the new tier and
confirm.

**Effect on streams and quality**

Simultaneous stream limits and maximum resolution follow the plan. See
"Simultaneous streams by plan" and "Video quality settings" for the
limits that apply to each tier.
""",
    },
    {
        "article_id": "bill-008",
        "title": "Promotional pricing and when it ends",
        "category": "billing",
        "last_updated": "2025-12-15",
        "body": """
Promotional rates are temporary discounts applied to your subscription
for a fixed number of billing cycles.

**How promotions work**

- The discount is applied automatically at checkout when you use a valid
  promotional link or code
- The promotional rate holds for the stated number of cycles, typically
  3 or 6 months
- After the final promotional cycle, billing reverts to the standard
  rate for your plan

**Notification before the rate changes**

We email you 7 days before the first full-price charge. The invoice for
the final promotional month also shows the upcoming standard rate.

**Checking your status**

Account → Membership shows "Promotional rate active" with the date of
the final discounted cycle if a promotion applies.

**Stacking**

Only one promotion can apply at a time. A new code does not extend or
combine with an active promotion.
""",
    },
    {
        "article_id": "bill-009",
        "title": "About the Basic Plus plan",
        "category": "billing",
        "last_updated": "2024-06-03",
        "body": """
Basic Plus sits between the Basic and Standard tiers and includes two
simultaneous streams at 1080p resolution.

**Basic Plus includes**

- 2 simultaneous streams
- 1080p maximum resolution
- Downloads on up to 2 devices
- $11 per month

**Comparison with adjacent tiers**

| | Basic | Basic Plus | Standard |
|---|---|---|---|
| Price | $9 | $11 | $14 |
| Streams | 1 | 2 | 2 |
| Resolution | 720p | 1080p | 1080p |

**Switching to Basic Plus**

Account → Membership → Change plan → Basic Plus. The change follows the
standard upgrade and downgrade rules.

**Availability**

Basic Plus is offered in the United States, Canada, and the United
Kingdom.
""",
    },
    {
        "article_id": "bill-010",
        "title": "Taxes and regional pricing",
        "category": "billing",
        "last_updated": "2026-02-28",
        "body": """
Advertised prices exclude tax. The amount charged includes any sales
tax, VAT, or GST required in your billing region.

**How tax is determined**

Tax is based on the billing address associated with your payment method,
not on your current location or IP address.

**Regional price differences**

Base subscription prices vary by country to reflect local market
conditions and currency. The plan tier structure — Basic, Standard,
Premium — is consistent everywhere, but the amounts differ.

**Moving countries**

Update your billing address in Account → Billing. Pricing adjusts to the
new region at your next billing date. Content catalogues also differ by
region; see "Regional availability".

**Currency**

You are billed in the currency of your billing region. If your card is
issued elsewhere, your bank may apply a conversion fee that StreamFlix
does not control.
""",
    },

    # =================================================================
    # STREAMING QUALITY
    # =================================================================
    {
        "article_id": "strm-001",
        "title": "Video quality settings",
        "category": "streaming",
        "last_updated": "2026-01-30",
        "body": """
StreamFlix adjusts video quality automatically based on your connection
speed, device, and plan.

**Maximum resolution by plan**

| Plan | Maximum resolution |
|---|---|
| Basic | 720p |
| Standard | 1080p |
| Premium | 4K UHD |

**Manual control**

Account → Playback settings lets you cap data usage per profile:

- **Auto** — balances quality against available bandwidth (default)
- **High** — always requests the maximum your plan allows
- **Medium** — approximately 0.7 GB per hour
- **Low** — approximately 0.3 GB per hour

Changes apply to the selected profile only and take effect on the next
title you start.

**Why quality drops mid-playback**

The player steps down resolution when it detects reduced throughput,
which prevents interruption. Quality recovers automatically when
conditions improve.
""",
    },
    {
        "article_id": "strm-002",
        "title": "Buffering and playback problems",
        "category": "streaming",
        "last_updated": "2026-03-02",
        "body": """
Buffering usually indicates a bandwidth or local network problem rather
than an issue with the StreamFlix service.

**Try these in order**

1. Restart the StreamFlix app
2. Restart your device
3. Restart your router and modem, waiting 30 seconds before powering on
4. Move closer to the router or switch to a wired connection
5. Pause other high-bandwidth activity on the network
6. Lower playback quality in Account → Playback settings

**Check your speed**

Run a speed test on the same device you are streaming from. Compare the
result against the requirements in "Bandwidth requirements by
resolution".

**If only one title buffers**

Try a different title. If others play normally, the problem is specific
to that title's encoding — report it through Help → Report a problem.

**Service status**

Check status.streamflix.example before troubleshooting further; a
regional incident produces symptoms identical to a local network fault.
""",
    },
    {
        "article_id": "strm-003",
        "title": "Bandwidth requirements by resolution",
        "category": "streaming",
        "last_updated": "2025-10-18",
        "body": """
Sustained download speed needed per concurrent stream.

| Resolution | Minimum | Recommended |
|---|---|---|
| 480p (SD) | 1.5 Mbps | 3 Mbps |
| 720p (HD) | 3 Mbps | 5 Mbps |
| 1080p (Full HD) | 5 Mbps | 8 Mbps |
| 4K UHD | 15 Mbps | 25 Mbps |

**Multiple streams**

Requirements are per stream. Two 1080p streams need roughly 16 Mbps of
sustained capacity, not 8.

**Why "sustained" matters**

Peak speeds reported by speed tests are not the same as sustained
throughput. A connection that peaks at 50 Mbps but drops to 4 Mbps
during congestion will buffer at 1080p.

**Data consumption**

| Quality | Approx. per hour |
|---|---|
| Low | 0.3 GB |
| Medium | 0.7 GB |
| High (1080p) | 3 GB |
| High (4K) | 7 GB |
""",
    },
    {
        "article_id": "strm-004",
        "title": "Audio problems",
        "category": "streaming",
        "last_updated": "2025-09-22",
        "body": """
**No sound**

- Confirm the device is not muted and volume is raised on both the app
  and the device itself
- Check the audio track selected in the player — some titles default to
  a language track you may not expect
- On external speakers or soundbars, confirm the correct input is active
- Restart the app

**Audio out of sync with video**

Usually caused by the device's audio processing, not the stream. Restart
the title first. If it persists, disable any audio post-processing or
"night mode" on your TV or receiver.

**Audio track or language missing**

Available tracks vary by title and region. If a title offered a language
previously and no longer does, the licensing terms for that region
changed.

**Distorted or low-volume audio**

Some titles are mastered with a wide dynamic range. Enabling a
compression or "clear voice" mode on your device raises dialogue
relative to effects.
""",
    },
    {
        "article_id": "strm-005",
        "title": "Subtitles and captions",
        "category": "streaming",
        "last_updated": "2025-11-05",
        "body": """
**Turning subtitles on**

Select the speech-bubble icon in the player and choose a subtitle track.
The choice persists for the profile until changed.

**Styling subtitles**

Account → Subtitle appearance lets you set font, size, colour, and
background opacity per profile. A preview updates as you adjust.

**Subtitles vs closed captions**

Subtitles render dialogue only. Closed captions also describe relevant
non-speech audio and are labelled "CC" in the track list. Not every
title offers CC.

**Missing languages**

Subtitle availability is set per title and per region by the licensor.
A language available in one country may be absent in another for the
same title.

**Subtitles out of sync**

Restart the title. If the offset persists across restarts and devices,
report it through Help → Report a problem and include the title and
timestamp.
""",
    },
    {
        "article_id": "strm-006",
        "title": "Downloads for offline viewing",
        "category": "streaming",
        "last_updated": "2026-01-12",
        "body": """
Downloads are available on the mobile apps for iOS and Android and on
Windows tablets. Downloads are not supported in web browsers or on
smart-TV apps.

**Device limits by plan**

| Plan | Devices with downloads |
|---|---|
| Basic | 1 |
| Standard | 2 |
| Premium | 4 |

**Downloading**

Open a title and select the download icon. Not every title is
downloadable — availability is set by the licensor and shown by the
presence of the icon.

**Expiry**

Most downloads expire 30 days after download, or 48 hours after you
start watching, whichever comes first. Some titles have shorter windows
shown on the download itself.

**Storage**

Downloads count against device storage. Choose Standard or High download
quality in app settings to trade file size against resolution.
""",
    },
    {
        "article_id": "strm-007",
        "title": "Simultaneous streams by plan",
        "category": "streaming",
        "last_updated": "2026-01-08",
        "body": """
The number of devices that can stream at the same time is set by your
plan.

| Plan | Simultaneous streams |
|---|---|
| Basic | 1 |
| Standard | 2 |
| Premium | 4 |

**"Too many devices" error**

This appears when a new stream would exceed the plan limit. Stop
playback on another device, or upgrade the plan.

A stream can remain counted for a few minutes after a device is
switched off without stopping playback cleanly. Waiting five minutes
usually clears it.

**Profiles vs streams**

Every plan supports up to 5 profiles. Profiles and simultaneous streams
are separate limits — 5 profiles exist on a Basic plan, but only one can
stream at any moment.

**Downloads**

Playing a downloaded title offline does not consume a simultaneous
stream.
""",
    },
    {
        "article_id": "strm-008",
        "title": "HDR and Dolby Atmos",
        "category": "streaming",
        "last_updated": "2025-12-01",
        "body": """
**HDR**

High dynamic range is available on the Premium plan only, and requires
an HDR-capable display and a title mastered in HDR. Supported formats
are HDR10 and Dolby Vision. Eligible titles show an HDR badge on the
details page.

**Dolby Atmos**

Atmos requires the Premium plan, an Atmos-capable audio setup, and a
title mastered in Atmos. Eligible titles show an Atmos badge.

**Not seeing HDR or Atmos**

- Confirm the plan is Premium
- Confirm the title carries the badge
- On TVs, enable the enhanced-format setting for the HDMI input in use;
  many TVs disable it by default
- Confirm bandwidth is sufficient for 4K — HDR streams fall back to SDR
  when throughput drops

**Devices**

Support varies by device model and firmware. The device's own
specification is authoritative.
""",
    },

    # =================================================================
    # ACCOUNT MANAGEMENT
    # =================================================================
    {
        "article_id": "acct-001",
        "title": "Resetting your password",
        "category": "account",
        "last_updated": "2026-02-05",
        "body": """
**If you know your current password**

Account → Security → Change password. You will be asked for the current
password before setting a new one.

**If you have forgotten it**

1. Select **Forgot password** on the sign-in screen
2. Enter the email address on the account
3. Follow the link in the email we send

The reset link expires after 60 minutes. Requesting a new link
invalidates any earlier one.

**Not receiving the email**

- Check spam and promotions folders
- Confirm you entered the address the account was created with
- Allow up to 10 minutes
- Add no-reply@streamflix.example to your contacts

**Password requirements**

At least 8 characters, including one number. Passwords reused from a
known breach are rejected.

**After a reset**

All devices are signed out. Sign in again with the new password.
""",
    },
    {
        "article_id": "acct-002",
        "title": "Changing your email address",
        "category": "account",
        "last_updated": "2025-10-09",
        "body": """
**Steps**

1. Account → Personal details → Email
2. Enter the new address and your current password
3. Confirm through the verification link sent to the new address

The change completes only after verification. Until then, the original
address remains active.

**Both addresses are notified**

We email the previous address to confirm the change. If you receive that
notice and did not request the change, follow the "secure your account"
link in it immediately.

**Effect on your subscription**

Changing the email does not alter billing, plan, profiles, or viewing
history.

**If you have lost access to the old address**

Verification goes to the new address, so a lost old mailbox does not
block the change — but you must still be able to sign in. If you cannot
sign in, reset the password first.
""",
    },
    {
        "article_id": "acct-003",
        "title": "Managing profiles",
        "category": "account",
        "last_updated": "2026-01-25",
        "body": """
Every StreamFlix account supports up to 5 profiles, on all plans.
Profiles keep viewing history, recommendations, My List, and playback
settings separate.

**Adding a profile**

From the profile selection screen choose **Add profile**, enter a name,
and pick whether it is a kids profile.

**Editing**

Manage profiles → select a profile to rename it, change the avatar, set
the language, or adjust maturity settings.

**Deleting**

Deleting a profile permanently removes its viewing history, My List, and
recommendations. This cannot be undone. The primary account profile
cannot be deleted.

**Profile locks**

Set a 4-digit PIN on any profile in Manage profiles → Profile lock. The
PIN is required each time that profile is selected.

**Profiles and streams**

Profiles do not increase simultaneous stream limits — those follow the
plan. See "Simultaneous streams by plan".
""",
    },
    {
        "article_id": "acct-004",
        "title": "Kids profiles and parental controls",
        "category": "account",
        "last_updated": "2026-01-25",
        "body": """
**Kids profiles**

A kids profile shows only titles rated for children, uses a simplified
interface, and hides account and billing settings entirely.

Create one from Add profile by enabling the **Kids** toggle.

**Maturity ratings**

Manage profiles → select the profile → Maturity settings. Choose the
highest rating that profile may view. Ratings follow the classification
system of your region.

**Title-level blocks**

Block specific titles regardless of rating in the same Maturity settings
screen. Blocked titles do not appear in search or recommendations for
that profile.

**Protecting the settings**

Set an account PIN in Account → Parental controls. The PIN is required
to change maturity settings or create a non-kids profile, which prevents
a child from raising their own limits.

**Viewing activity**

Account → Viewing activity shows what each profile watched and when.
""",
    },
    {
        "article_id": "acct-005",
        "title": "Viewing history and privacy",
        "category": "account",
        "last_updated": "2025-11-19",
        "body": """
**Reviewing history**

Account → Viewing activity, then choose a profile. Entries show title,
date, and device.

**Removing an entry**

Select the hide icon beside any entry. Hidden titles stop influencing
that profile's recommendations within about 24 hours. Hiding an episode
offers the option to hide the whole series.

**What recommendations use**

Titles watched, how much of each was watched, ratings given, and My List
contents — all scoped to the individual profile, never pooled across
profiles.

**Data retention**

Viewing history is retained while the account is active and for 10
months after cancellation, which is what allows history to be restored
if you resubscribe.

**Devices signed in**

Account → Security → Recent device activity lists sign-ins. **Sign out
of all devices** ends every active session.
""",
    },
    {
        "article_id": "acct-006",
        "title": "Deleting your account",
        "category": "account",
        "last_updated": "2025-12-08",
        "body": """
Deleting an account is permanent and distinct from cancelling a
subscription.

**Cancel vs delete**

- **Cancel** — billing stops, access continues to period end, history is
  retained for 10 months, resubscribing restores everything
- **Delete** — the account and all associated data are removed and
  cannot be recovered

**Before deleting**

Cancel any active subscription first. Deleting does not automatically
stop billing on a subscription purchased through a third party such as
an app store — cancel that separately with the provider.

**How to delete**

Account → Personal details → Delete account. You will be asked to
confirm with your password.

**Timing**

Deletion completes within 30 days. During that window, signing in
cancels the deletion.
""",
    },
    {
        "article_id": "acct-007",
        "title": "Suspicious account activity",
        "category": "account",
        "last_updated": "2026-02-20",
        "body": """
**Signs of unauthorised access**

- Profiles you did not create
- Titles in viewing activity you did not watch
- Plan or billing changes you did not make
- Sign-in notifications from unfamiliar locations

**Act in this order**

1. Change your password — Account → Security → Change password
2. **Sign out of all devices** — Account → Security → Recent device
   activity
3. Review Account → Billing for charges you do not recognise
4. Delete any profiles you did not create
5. Enable two-factor authentication

**Unrecognised charges**

Report them through Help → Contact us. Charges resulting from
unauthorised access are eligible for refund; see "Refund policy".

**Preventing recurrence**

Use a password unique to StreamFlix. Reused passwords exposed in
breaches elsewhere are the most common cause of account takeover.
""",
    },
    {
        "article_id": "acct-008",
        "title": "Two-factor authentication",
        "category": "account",
        "last_updated": "2026-02-20",
        "body": """
Two-factor authentication adds a one-time code to sign-in, so a stolen
password alone is not enough to access the account.

**Enabling**

Account → Security → Two-factor authentication. Choose:

- **Authenticator app** — recommended; scan the QR code with any TOTP
  app
- **SMS** — codes sent to a verified mobile number

**Recovery codes**

Enabling generates 10 single-use recovery codes. Store them outside the
authenticator device. Without a recovery code, losing the device means
contacting support and completing identity verification.

**When a code is requested**

On sign-in from a new device, and on sensitive changes such as email or
payment method. Existing signed-in devices are unaffected.

**Turning it off**

The same screen, after entering a current code.
""",
    },

    # =================================================================
    # DEVICES
    # =================================================================
    {
        "article_id": "dev-001",
        "title": "Supported devices",
        "category": "devices",
        "last_updated": "2026-03-01",
        "body": """
**Smart TVs**

Samsung (2018 and later), LG webOS (2018 and later), Sony and other
Android TV models, Vizio SmartCast, Hisense and TCL Roku TVs.

**Streaming devices**

Roku, Amazon Fire TV, Apple TV (4th generation and later), Chromecast,
Nvidia Shield.

**Mobile and tablet**

iOS 15 and later, Android 9 and later.

**Computers**

Chrome, Firefox, Safari, and Edge — see "Browser requirements" for
version and resolution limits.

**Game consoles**

PlayStation 4 and 5, Xbox One and Series X|S.

**Not supported**

Windows Phone, devices running Android below 9, smart TVs from before
2018, and rooted or jailbroken devices.

**Resolution by device**

4K requires a Premium plan and a device that supports 4K output. Many
older devices cap at 1080p regardless of plan.
""",
    },
    {
        "article_id": "dev-002",
        "title": "Installing StreamFlix on a smart TV",
        "category": "devices",
        "last_updated": "2026-02-14",
        "body": """
**General steps**

1. Open the TV's app store — Samsung Apps, LG Content Store, Google Play
   Store on Android TV, or the Vizio app row
2. Search for StreamFlix
3. Select Install
4. Open the app and sign in

**Signing in with a code**

Most TVs offer code-based sign-in. The app displays a code; enter it at
streamflix.example/tv on a phone or computer. This avoids typing a
password with a remote.

**App missing from the store**

The TV model may predate support. Check the manufacturer and year
against "Supported devices". A streaming stick is the usual workaround
for an unsupported TV.

**App installed but will not open**

Update the TV firmware, then reinstall the app. If it still fails, a
factory reset of the TV resolves most remaining cases.
""",
    },
    {
        "article_id": "dev-003",
        "title": "Setting up StreamFlix on Roku",
        "category": "devices",
        "last_updated": "2026-02-14",
        "body": """
**Adding the channel**

1. Press Home on the Roku remote
2. Select **Streaming Channels**
3. Search for StreamFlix
4. Select **Add Channel**
5. Return Home and open StreamFlix

**Signing in**

The channel shows an activation code. Enter it at
streamflix.example/tv while signed in on another device.

**Requirements**

Roku OS 9.4 or later. Update through Settings → System → System update.

**4K playback**

Requires a 4K-capable Roku model, a Premium plan, and an HDMI 2.0 input
on the TV. Roku Express and older Streaming Stick models output 1080p
maximum.

**Playback problems**

Remove the channel, restart the Roku from Settings → System → Power →
System restart, then add the channel again. This clears cached data that
a simple restart leaves in place.
""",
    },
    {
        "article_id": "dev-004",
        "title": "Setting up StreamFlix on Apple TV",
        "category": "devices",
        "last_updated": "2026-02-14",
        "body": """
**Installing**

1. Open the App Store on Apple TV
2. Search for StreamFlix
3. Select **Get**
4. Open the app and sign in

**Signing in**

Sign in directly with the on-screen keyboard, or use the activation-code
flow at streamflix.example/tv.

**Requirements**

Apple TV 4th generation or later, running tvOS 15 or later. Older Apple
TV models are not supported.

**4K and HDR**

Apple TV 4K with a Premium plan supports 4K and Dolby Vision. Enable the
matching format in Settings → Video and Audio → Format.

**Playback problems**

Force-quit the app by double-pressing TV and swiping up, then reopen.
If problems persist, delete and reinstall the app — this clears the
local cache.
""",
    },
    {
        "article_id": "dev-005",
        "title": "Setting up StreamFlix on Fire TV",
        "category": "devices",
        "last_updated": "2026-02-14",
        "body": """
**Installing**

1. From the Fire TV home screen, select the search icon
2. Search for StreamFlix
3. Select **Get** or **Download**
4. Open the app and sign in

**Signing in**

Use the activation code shown in the app at streamflix.example/tv, or
type credentials directly with the remote.

**Requirements**

Fire OS 6 or later. Fire TV Stick (2nd generation and later), Fire TV
Cube, and Fire TV Edition televisions are supported.

**4K playback**

Requires Fire TV Stick 4K, Fire TV Stick 4K Max, or Fire TV Cube, plus a
Premium plan. The basic Fire TV Stick outputs 1080p maximum.

**Playback problems**

Settings → Applications → Manage Installed Applications → StreamFlix →
Clear cache. Clear data as a second step; this signs you out and
requires signing in again.
""",
    },
    {
        "article_id": "dev-006",
        "title": "StreamFlix on mobile devices",
        "category": "devices",
        "last_updated": "2026-01-18",
        "body": """
**Installing**

Download from the App Store on iOS or Google Play on Android. iOS 15 or
later and Android 9 or later are required.

**Downloads**

The mobile apps support offline downloads. Device limits follow the
plan; see "Downloads for offline viewing".

**Mobile data**

App settings → Video playback lets you restrict streaming to Wi-Fi or
cap cellular quality. Default behaviour is "Automatic", which uses less
data on cellular than on Wi-Fi.

**Casting**

Cast to a Chromecast or AirPlay device using the cast icon in the
player. See "Casting and AirPlay".

**Picture-in-picture**

Supported on iOS 15 and later and on most Android 10 and later devices.
Swipe up during playback to continue watching in a floating window.

**Battery**

Downloading and watching offline consumes less battery than streaming,
because the radio stays idle.
""",
    },
    {
        "article_id": "dev-007",
        "title": "Browser requirements",
        "category": "devices",
        "last_updated": "2025-12-20",
        "body": """
**Supported browsers and maximum resolution**

| Browser | Maximum resolution |
|---|---|
| Google Chrome | 1080p |
| Mozilla Firefox | 1080p |
| Microsoft Edge | 4K (Windows 10+ with HEVC support) |
| Safari | 4K (macOS 11+ on compatible Macs) |

Chrome and Firefox are capped at 1080p by the digital-rights
restrictions those browsers enforce, not by StreamFlix.

**Requirements**

- A current browser version — we support the two most recent major
  releases
- JavaScript and cookies enabled
- Hardware acceleration enabled for smooth 1080p playback

**No downloads in browsers**

Offline downloads are unavailable on the web. Use the mobile or Windows
tablet apps.

**Playback errors in-browser**

Clear the browser cache, disable extensions — ad blockers and privacy
extensions are the most common cause — and confirm the operating system
is current.
""",
    },
    {
        "article_id": "dev-008",
        "title": "Casting and AirPlay",
        "category": "devices",
        "last_updated": "2025-11-11",
        "body": """
**Chromecast**

Confirm the phone and Chromecast are on the same Wi-Fi network, then
select the cast icon in the StreamFlix player and choose the device.
Playback moves to the TV and the phone becomes the remote.

**AirPlay**

Select the AirPlay icon in the player and choose an Apple TV or
AirPlay-capable TV. Requires iOS 15 or later.

**Casting stops or fails to start**

- Confirm both devices are on the same network — guest networks and
  band-separated 2.4/5 GHz SSIDs are a frequent cause
- Restart the receiving device
- Update the StreamFlix app
- Disable VPNs on the phone

**Quality while casting**

Resolution follows the plan and the receiving device, not the phone. A
Basic plan casts at 720p even to a 4K TV.

**Stream count**

A cast stream counts as one simultaneous stream.
""",
    },
    {
        "article_id": "dev-009",
        "title": "StreamFlix on game consoles",
        "category": "devices",
        "last_updated": "2025-10-30",
        "body": """
**Supported consoles**

PlayStation 4, PlayStation 5, Xbox One, Xbox Series X and Series S.

**Installing**

Find StreamFlix in the console's store — PlayStation Store or Microsoft
Store — and install it. Sign in with the activation-code flow at
streamflix.example/tv.

**4K**

PlayStation 5, Xbox Series X, and Xbox One S/X support 4K output with a
Premium plan and a 4K display. PlayStation 4 and Xbox One (original)
output 1080p maximum.

**Playback problems**

Close the application fully rather than suspending it, then reopen.
Suspended sessions are a common source of playback errors after the
console wakes from rest mode.

**Older consoles**

PlayStation 3 and Xbox 360 are no longer supported. The apps were
retired in 2024 and will not install or sign in.
""",
    },

    # =================================================================
    # CONTENT & CATALOGUE
    # =================================================================
    {
        "article_id": "cont-001",
        "title": "Why titles leave StreamFlix",
        "category": "content",
        "last_updated": "2025-09-14",
        "body": """
Titles StreamFlix does not own are licensed for a fixed term. When a
licence ends and is not renewed, the title leaves the catalogue.

**Why a licence may not be renewed**

- The rights holder moved the title to their own service
- Renewal terms were not commercially viable
- The rights holder withdrew it from licensing entirely

**Advance notice**

Titles expiring within 30 days show a "Last day to watch" badge on the
details page. Anything saved to My List that is expiring soon appears in
the "Leaving soon" row.

**Downloads**

A downloaded title becomes unplayable once its licence ends, even if the
download has not expired.

**Requesting a title**

Use Help → Suggest a title. We cannot commit to acquiring specific
titles, and we cannot say whether a departed title will return.

**StreamFlix Originals**

Originals are owned outright and do not expire.
""",
    },
    {
        "article_id": "cont-002",
        "title": "Regional content availability",
        "category": "content",
        "last_updated": "2025-12-05",
        "body": """
Catalogues differ by country because licensing is negotiated
region-by-region. A title available in one country may be absent in
another, and StreamFlix Originals are the main exception — they are
available everywhere the service operates.

**Which catalogue you see**

Determined by your current location, not your billing address. Travelling
shows you the local catalogue for the country you are in.

**Travelling**

Your account works in any country where StreamFlix operates. Downloads
made before travelling remain playable while their licence lasts, which
is often the most practical option.

**VPNs and proxies**

Accessing StreamFlix through a VPN or proxy generally restricts you to
Originals only, because we cannot verify your region. This is a
licensing obligation.

**Moving permanently**

Update the billing address; pricing and catalogue both follow.
""",
    },
    {
        "article_id": "cont-003",
        "title": "New releases and when they arrive",
        "category": "content",
        "last_updated": "2026-02-01",
        "body": """
**Where to find new titles**

The "New & Popular" row on the home screen, refreshed daily. Full-season
Originals are added at 00:01 UTC on their release date.

**Weekly episode releases**

Some series release weekly rather than all at once. The details page
shows the next episode's date when a weekly schedule applies.

**Notifications**

Enable release notifications for a title from its details page. Mobile
notifications must also be enabled at the operating-system level.

**Coming soon**

The "Coming soon" row lists titles arriving within 30 days. Selecting
**Remind me** sends a notification on release day.

**Release-date differences by region**

Licensing terms sometimes stagger release dates between countries.
Originals release simultaneously worldwide.
""",
    },
    {
        "article_id": "cont-004",
        "title": "Search and recommendations",
        "category": "content",
        "last_updated": "2026-01-16",
        "body": """
**Search**

Search by title, actor, director, or genre. Results are limited to
titles available in your current region.

**How recommendations are built**

Per profile, from titles watched and how much of each was watched,
ratings given, My List contents, time of day and device, and the
behaviour of viewers with similar patterns.

**Improving them**

- Rate titles with thumbs up or down
- Hide titles you would rather not influence suggestions — Account →
  Viewing activity
- Use separate profiles for genuinely different tastes; a shared profile
  produces recommendations that suit nobody

**Why a title you disliked keeps appearing**

Rating alone is a weak signal. Hiding it from viewing activity removes
it from the recommendation inputs entirely.

**Row ordering**

Rows and their order are personalised. Two profiles on the same account
will see different home screens.
""",
    },
    {
        "article_id": "cont-005",
        "title": "My List",
        "category": "content",
        "last_updated": "2025-11-27",
        "body": """
**Adding titles**

Select the **+** icon on any title card or details page. My List is
per-profile, not shared across the account.

**Finding it**

The My List row on the home screen, or the My List entry in the main
navigation.

**Ordering**

Manual ordering is available on the web and mobile apps. Smart TV apps
display My List in the order titles were added.

**Capacity**

Up to 500 titles per profile.

**Titles disappearing from My List**

A title removed from the catalogue also leaves My List. Items expiring
within 30 days appear in "Leaving soon" before they go.

**Effect on recommendations**

My List contents feed that profile's recommendations, so adding titles
you intend to watch improves suggestions.
""",
    },
    {
        "article_id": "cont-006",
        "title": "Content ratings explained",
        "category": "content",
        "last_updated": "2025-10-22",
        "body": """
Ratings follow the classification system of your region — TV Parental
Guidelines and MPA in the United States, BBFC in the United Kingdom, and
the relevant national body elsewhere.

**Where ratings appear**

On the details page, and briefly on screen when playback begins.

**Maturity settings**

Each profile has a maximum rating, set in Manage profiles → Maturity
settings and protected by the account PIN. See "Kids profiles and
parental controls".

**Content advisories**

Beneath the rating, short advisories describe why a title carries it —
for example "violence, strong language".

**Ratings differing between regions**

The same title may carry different ratings in different countries
because each classification body applies its own criteria.

**Unrated titles**

A small number of titles have no regional classification. These are
treated as the highest maturity level and hidden from kids profiles.
""",
    },

    # =================================================================
    # TRIAL & ONBOARDING
    # =================================================================
    {
        "article_id": "trial-001",
        "title": "How the 14-day free trial works",
        "category": "trial",
        "last_updated": "2026-01-05",
        "body": """
New subscribers receive 14 days of full access at no charge.

**What the trial includes**

Full catalogue access at the plan tier selected at sign-up, including
downloads and all profile features. The trial is not a restricted tier.

**Payment method**

A valid payment method is required at sign-up. Nothing is charged during
the trial; some banks show a temporary authorisation hold of a small
amount, which is released within a few days.

**When the trial ends**

The first charge is taken on day 15 at the standard rate for the plan
selected. See "What happens when your trial ends".

**Tracking the time remaining**

Account → Membership shows the trial end date.

**Changing plan during the trial**

Switching tiers mid-trial is allowed and does not extend or restart the
trial period. The new tier's rate applies at the first charge.
""",
    },
    {
        "article_id": "trial-002",
        "title": "What happens when your trial ends",
        "category": "trial",
        "last_updated": "2026-01-05",
        "body": """
**Automatic conversion**

Unless cancelled first, the subscription converts to a paid plan on day
15 and the payment method on file is charged the standard rate for the
selected tier.

**Reminder**

We email you 3 days before the trial ends, stating the date and the
amount of the first charge.

**Avoiding the charge**

Cancel any time before day 15 — see "Canceling during your trial".
Access continues to the end of the trial period.

**If the first charge fails**

The standard failed-payment retry schedule applies; see "Failed or
declined payments". Access continues during the retry window.

**Continuity**

Profiles, viewing history, My List, and downloads carry over unchanged.
Nothing is reset at conversion.
""",
    },
    {
        "article_id": "trial-003",
        "title": "Free trial eligibility",
        "category": "trial",
        "last_updated": "2025-11-02",
        "body": """
The free trial is available once per household.

**Who is eligible**

New subscribers who have not previously held a StreamFlix subscription
or trial. Eligibility is assessed against payment method and household
address, not the email address alone — a new email does not create a new
eligibility.

**Not eligible**

- Anyone who has previously used a trial
- Returning subscribers, including those whose account was cancelled
  years ago
- Accounts created through certain partner or bundle offers, which carry
  their own terms

**Checking before signing up**

The sign-up flow states whether a trial applies before you confirm. If
no trial banner appears, the account is not eligible and billing starts
immediately.

**Partner bundles**

Subscriptions obtained through a mobile carrier or ISP bundle follow the
partner's promotional terms instead of the standard trial.
""",
    },
    {
        "article_id": "trial-004",
        "title": "Canceling during your trial",
        "category": "trial",
        "last_updated": "2026-01-05",
        "body": """
Cancelling before day 15 means no charge is made.

**How**

1. Account → Membership
2. Select **Cancel membership**
3. Confirm

A confirmation email follows. Without that email, the cancellation did
not complete.

**Access after cancelling**

Full access continues to the end of the 14-day period. Cancelling on day
3 keeps access through day 14.

**If you are charged after cancelling**

The cancellation likely did not complete. Charges taken after a
confirmed cancellation are eligible for refund; see "Refund policy".

**Resubscribing later**

You can resubscribe at any time, but the trial is not offered again —
see "Free trial eligibility". Profiles and history are retained for 10
months.
""",
    },
]


# ---------------------------------------------------------------------
# Documented flaws — used by the Phase 1 audit and the evaluation phases
# ---------------------------------------------------------------------
KNOWN_CORPUS_FLAWS: list[dict] = [
    {
        "flaw_id": "contradiction-refund-window",
        "kind": "contradiction",
        "article_ids": ["bill-002", "bill-003"],
        "description": (
            "bill-002 states a 30-day refund window; bill-003 states 14 days. "
            "A correct system should surface the conflict or hedge, not assert "
            "one figure confidently."
        ),
    },
    {
        "flaw_id": "outdated-basic-plus",
        "kind": "outdated",
        "article_ids": ["bill-009"],
        "description": (
            "Documents the Basic Plus tier, discontinued in 2025 and absent "
            "from the current pricing table in bill-001. Last updated 2024-06-03."
        ),
    },
    {
        "flaw_id": "near-duplicate-device-setup",
        "kind": "near_duplicate",
        "article_ids": ["dev-003", "dev-004", "dev-005"],
        "description": (
            "Roku / Apple TV / Fire TV setup articles are structurally "
            "near-identical. Tests whether retrieval returns the right sibling "
            "rather than a plausible neighbour."
        ),
    },
    {
        "flaw_id": "near-duplicate-cancellation",
        "kind": "near_duplicate",
        "article_ids": ["bill-003", "trial-004"],
        "description": (
            "DISCOVERED, NOT PLANTED. Surfaced by the Phase 1 similarity "
            "analysis as the corpus's most lexically similar pair (~32% "
            "Jaccard), above the planted device cluster. Both describe the "
            "same cancellation flow for different account states — paid "
            "subscription vs active trial — and differ mainly in the "
            "consequences, which is exactly what a user asking 'how do I "
            "cancel' needs disambiguated. Probed by am-009."
        ),
    },
]

# Topics deliberately absent from the corpus. Questions on these must be
# refused, not answered — this is the guardrail the eval phase measures.
COVERAGE_GAPS: list[str] = [
    "gift subscriptions and gift cards",
    "business, corporate, and educational accounts",
    "accessibility features (screen readers, audio description)",
    "live sports and live event streaming",
    "password-sharing enforcement and extra-member fees",
    "GDPR data export and subject access requests",
    "affiliate and partner programmes",
    "physical merchandise and DVDs",
]
