"""Golden evaluation set — 120 questions with ground-truth labels.

This file is the load-bearing artifact of the whole project. Every
retrieval and generation metric downstream is measured against these
labels, so the evaluation is only ever as good as this set.

Four deliberate question categories:

    single_hop    (60)  Answer lives in exactly one article. Baseline
                        retrieval and faithfulness.
    multi_hop     (20)  Answer requires two or more articles. Tests
                        context assembly, not just top-1 retrieval.
    ambiguous     (15)  Several articles are plausibly relevant. Tests
                        the precision/recall trade-off directly.
    out_of_scope  (25)  No article covers the topic. The system MUST
                        refuse. This is the hallucination guardrail.

Fields:
    question_id           stable identifier
    question              user-phrased query, deliberately not keyword-matched
                          to article titles
    category              one of the four above
    gt_article_ids        ground-truth source articles; empty for out_of_scope
    reference_answer      what a correct answer contains; the judge scores
                          against this
    notes                 why this question earns its place in the set
"""
from __future__ import annotations

GOLDEN_QUESTIONS: list[dict] = [

    # =================================================================
    # SINGLE-HOP  (60)
    # =================================================================

    # --- billing ---
    {"question_id": "sh-001",
     "question": "When exactly does StreamFlix take my monthly payment?",
     "category": "single_hop", "gt_article_ids": ["bill-001"],
     "reference_answer": "On the same calendar day each month, set by the date you first subscribed. Months with fewer days bill on the final day. Payment is taken in advance for the coming month.",
     "notes": ""},

    {"question_id": "sh-002",
     "question": "How much does the Premium plan cost and how many people can watch at once?",
     "category": "single_hop", "gt_article_ids": ["bill-001"],
     "reference_answer": "$19 per month excluding tax, with 4 simultaneous streams.",
     "notes": ""},

    {"question_id": "sh-003",
     "question": "My card was declined. What happens to my account now?",
     "category": "single_hop", "gt_article_ids": ["bill-004"],
     "reference_answer": "StreamFlix retries up to four times over eight days (billing date, then days 3, 5, 8). The account stays active during that window. Access is suspended on day 9 if all retries fail; profiles and history are preserved and access resumes once a valid payment method is added.",
     "notes": ""},

    {"question_id": "sh-004",
     "question": "Can I pay with a prepaid card?",
     "category": "single_hop", "gt_article_ids": ["bill-005"],
     "reference_answer": "No. Prepaid cards are not accepted, nor are virtual card numbers expiring before the next billing date or direct bank transfer. Visa, Mastercard, Amex, qualifying debit cards, and PayPal in supported regions are accepted.",
     "notes": "Negative answer — tests whether the system states the exclusion rather than hedging."},

    {"question_id": "sh-005",
     "question": "Why is my bill different this month than last month?",
     "category": "single_hop", "gt_article_ids": ["bill-006"],
     "reference_answer": "Common causes: a promotional rate ended, a mid-cycle plan change produced a prorated line, a local tax rate changed, or currency conversion applied because the card is issued outside the billing region.",
     "notes": ""},

    {"question_id": "sh-006",
     "question": "If I upgrade my plan today, when does it take effect?",
     "category": "single_hop", "gt_article_ids": ["bill-007"],
     "reference_answer": "Immediately. You are charged a prorated amount for the rest of the current cycle and the new rate applies from the next billing date. Extra streams and higher resolution become available right away.",
     "notes": ""},

    {"question_id": "sh-007",
     "question": "I want to move to a cheaper plan — do I get money back for the rest of the month?",
     "category": "single_hop", "gt_article_ids": ["bill-007"],
     "reference_answer": "No. Downgrades take effect at the end of the current billing period, so you keep the higher tier for time already paid for. No proration or partial refund is issued.",
     "notes": ""},

    {"question_id": "sh-008",
     "question": "Will you tell me before my discount runs out?",
     "category": "single_hop", "gt_article_ids": ["bill-008"],
     "reference_answer": "Yes. StreamFlix emails 7 days before the first full-price charge, and the final promotional invoice also shows the upcoming standard rate.",
     "notes": ""},

    {"question_id": "sh-009",
     "question": "Can I use two promo codes at the same time?",
     "category": "single_hop", "gt_article_ids": ["bill-008"],
     "reference_answer": "No. Only one promotion can apply at a time, and a new code does not extend or combine with an active promotion.",
     "notes": ""},

    {"question_id": "sh-010",
     "question": "Is the price I see on the website what I actually get charged?",
     "category": "single_hop", "gt_article_ids": ["bill-010"],
     "reference_answer": "No — advertised prices exclude tax. The charge includes sales tax, VAT, or GST as required, determined by the billing address on the payment method rather than current location.",
     "notes": ""},

    {"question_id": "sh-011",
     "question": "I'm moving to another country. What happens to my subscription price?",
     "category": "single_hop", "gt_article_ids": ["bill-010"],
     "reference_answer": "Update the billing address in Account → Billing; pricing adjusts to the new region at the next billing date. Base prices vary by country.",
     "notes": ""},

    {"question_id": "sh-012",
     "question": "How far back can I download my invoices?",
     "category": "single_hop", "gt_article_ids": ["bill-006"],
     "reference_answer": "Invoices are retained for 24 months and can be downloaded as PDFs from Account → Billing → Payment history.",
     "notes": ""},

    # --- streaming ---
    {"question_id": "sh-013",
     "question": "What internet speed do I need for 4K?",
     "category": "single_hop", "gt_article_ids": ["strm-003"],
     "reference_answer": "15 Mbps minimum, 25 Mbps recommended, sustained per stream.",
     "notes": ""},

    {"question_id": "sh-014",
     "question": "How much data does an hour of StreamFlix use on the low setting?",
     "category": "single_hop", "gt_article_ids": ["strm-003"],
     "reference_answer": "Approximately 0.3 GB per hour on Low.",
     "notes": ""},

    {"question_id": "sh-015",
     "question": "Why does the picture get blurry partway through a movie?",
     "category": "single_hop", "gt_article_ids": ["strm-001"],
     "reference_answer": "The player steps resolution down when it detects reduced throughput, which prevents an interruption. Quality recovers automatically when conditions improve.",
     "notes": ""},

    {"question_id": "sh-016",
     "question": "How do I stop StreamFlix eating my data allowance?",
     "category": "single_hop", "gt_article_ids": ["strm-001"],
     "reference_answer": "Account → Playback settings, then set the profile to Medium (~0.7 GB/hr) or Low (~0.3 GB/hr). The setting is per profile and applies to the next title started.",
     "notes": ""},

    {"question_id": "sh-017",
     "question": "It keeps buffering. What should I try first?",
     "category": "single_hop", "gt_article_ids": ["strm-002"],
     "reference_answer": "In order: restart the app, restart the device, restart the router and modem (waiting 30 seconds), move closer to the router or go wired, pause other high-bandwidth activity, and lower playback quality. Also check status.streamflix.example for a regional incident.",
     "notes": ""},

    {"question_id": "sh-018",
     "question": "Only one show buffers, everything else is fine. What does that mean?",
     "category": "single_hop", "gt_article_ids": ["strm-002"],
     "reference_answer": "If other titles play normally, the problem is specific to that title's encoding. Report it through Help → Report a problem.",
     "notes": ""},

    {"question_id": "sh-019",
     "question": "The sound doesn't match the actors' mouths.",
     "category": "single_hop", "gt_article_ids": ["strm-004"],
     "reference_answer": "Usually the device's audio processing rather than the stream. Restart the title first; if it persists, disable audio post-processing or night mode on the TV or receiver.",
     "notes": ""},

    {"question_id": "sh-020",
     "question": "Can I make the subtitle text bigger?",
     "category": "single_hop", "gt_article_ids": ["strm-005"],
     "reference_answer": "Yes — Account → Subtitle appearance lets you set font, size, colour, and background opacity per profile, with a live preview.",
     "notes": ""},

    {"question_id": "sh-021",
     "question": "What's the difference between subtitles and closed captions?",
     "category": "single_hop", "gt_article_ids": ["strm-005"],
     "reference_answer": "Subtitles render dialogue only; closed captions also describe relevant non-speech audio and are labelled CC. Not every title offers CC.",
     "notes": ""},

    {"question_id": "sh-022",
     "question": "How long do downloaded episodes last before they expire?",
     "category": "single_hop", "gt_article_ids": ["strm-006"],
     "reference_answer": "Most expire 30 days after download or 48 hours after you start watching, whichever comes first. Some titles have shorter windows shown on the download.",
     "notes": ""},

    {"question_id": "sh-023",
     "question": "Can I download shows to watch on my laptop browser?",
     "category": "single_hop", "gt_article_ids": ["strm-006"],
     "reference_answer": "No. Downloads work on the iOS and Android apps and on Windows tablets, but not in web browsers or on smart-TV apps.",
     "notes": "Negative answer."},

    {"question_id": "sh-024",
     "question": "I keep getting a 'too many devices' message.",
     "category": "single_hop", "gt_article_ids": ["strm-007"],
     "reference_answer": "A new stream would exceed the plan's simultaneous-stream limit. Stop playback elsewhere or upgrade. A stream can stay counted for a few minutes after a device is switched off without stopping cleanly — waiting five minutes usually clears it.",
     "notes": ""},

    {"question_id": "sh-025",
     "question": "Does watching something I downloaded count against my stream limit?",
     "category": "single_hop", "gt_article_ids": ["strm-007"],
     "reference_answer": "No. Playing a downloaded title offline does not consume a simultaneous stream.",
     "notes": ""},

    {"question_id": "sh-026",
     "question": "Why don't I get Dolby Atmos even though my soundbar supports it?",
     "category": "single_hop", "gt_article_ids": ["strm-008"],
     "reference_answer": "Atmos requires the Premium plan, an Atmos-capable setup, and a title mastered in Atmos (shown by an Atmos badge). Device support also varies by model and firmware.",
     "notes": ""},

    {"question_id": "sh-027",
     "question": "My HDR TV shows a normal picture on StreamFlix.",
     "category": "single_hop", "gt_article_ids": ["strm-008"],
     "reference_answer": "Check the plan is Premium, the title carries an HDR badge, the TV's enhanced-format setting is enabled for that HDMI input (often off by default), and bandwidth is sufficient — HDR falls back to SDR when throughput drops.",
     "notes": ""},

    # --- account ---
    {"question_id": "sh-028",
     "question": "How long is the password reset link good for?",
     "category": "single_hop", "gt_article_ids": ["acct-001"],
     "reference_answer": "60 minutes. Requesting a new link invalidates any earlier one.",
     "notes": ""},

    {"question_id": "sh-029",
     "question": "I asked for a password reset email and nothing arrived.",
     "category": "single_hop", "gt_article_ids": ["acct-001"],
     "reference_answer": "Check spam and promotions folders, confirm the address the account was created with, allow up to 10 minutes, and add no-reply@streamflix.example to contacts.",
     "notes": ""},

    {"question_id": "sh-030",
     "question": "What are the password rules?",
     "category": "single_hop", "gt_article_ids": ["acct-001"],
     "reference_answer": "At least 8 characters including one number. Passwords appearing in known breaches are rejected.",
     "notes": ""},

    {"question_id": "sh-031",
     "question": "If I change my email address, do I lose my watch history?",
     "category": "single_hop", "gt_article_ids": ["acct-002"],
     "reference_answer": "No. Changing the email does not alter billing, plan, profiles, or viewing history.",
     "notes": ""},

    {"question_id": "sh-032",
     "question": "How many profiles can I have?",
     "category": "single_hop", "gt_article_ids": ["acct-003"],
     "reference_answer": "Up to 5 profiles on every plan, regardless of tier.",
     "notes": ""},

    {"question_id": "sh-033",
     "question": "Can I put a PIN on just my own profile?",
     "category": "single_hop", "gt_article_ids": ["acct-003"],
     "reference_answer": "Yes — Manage profiles → Profile lock sets a 4-digit PIN on any profile, required each time that profile is selected.",
     "notes": ""},

    {"question_id": "sh-034",
     "question": "If I delete a profile can I get it back?",
     "category": "single_hop", "gt_article_ids": ["acct-003"],
     "reference_answer": "No. Deleting a profile permanently removes its viewing history, My List, and recommendations, and cannot be undone. The primary account profile cannot be deleted.",
     "notes": ""},

    {"question_id": "sh-035",
     "question": "How do I stop my kid changing their own maturity rating?",
     "category": "single_hop", "gt_article_ids": ["acct-004"],
     "reference_answer": "Set an account PIN in Account → Parental controls. The PIN is then required to change maturity settings or create a non-kids profile.",
     "notes": ""},

    {"question_id": "sh-036",
     "question": "Can I block one specific show from my child's profile?",
     "category": "single_hop", "gt_article_ids": ["acct-004"],
     "reference_answer": "Yes — title-level blocks are set in Maturity settings for that profile. Blocked titles do not appear in search or recommendations for it.",
     "notes": ""},

    {"question_id": "sh-037",
     "question": "How do I stop one embarrassing show from affecting my recommendations?",
     "category": "single_hop", "gt_article_ids": ["acct-005"],
     "reference_answer": "Account → Viewing activity, then select the hide icon beside the entry. Hidden titles stop influencing recommendations within about 24 hours; hiding an episode offers to hide the whole series.",
     "notes": ""},

    {"question_id": "sh-038",
     "question": "How do I see everywhere I'm currently signed in?",
     "category": "single_hop", "gt_article_ids": ["acct-005"],
     "reference_answer": "Account → Security → Recent device activity lists sign-ins, and 'Sign out of all devices' ends every active session.",
     "notes": ""},

    {"question_id": "sh-039",
     "question": "I clicked delete account by mistake — can I undo it?",
     "category": "single_hop", "gt_article_ids": ["acct-006"],
     "reference_answer": "Yes, within the 30-day completion window. Signing in during that period cancels the deletion.",
     "notes": ""},

    {"question_id": "sh-040",
     "question": "There are profiles on my account I didn't create. What do I do?",
     "category": "single_hop", "gt_article_ids": ["acct-007"],
     "reference_answer": "In order: change the password, sign out of all devices, review billing for unrecognised charges, delete the unfamiliar profiles, and enable two-factor authentication.",
     "notes": ""},

    {"question_id": "sh-041",
     "question": "What happens if I lose the phone with my authenticator app?",
     "category": "single_hop", "gt_article_ids": ["acct-008"],
     "reference_answer": "Use one of the 10 single-use recovery codes generated when 2FA was enabled. Without a recovery code you must contact support and complete identity verification.",
     "notes": ""},

    {"question_id": "sh-042",
     "question": "When will StreamFlix ask me for a 2FA code?",
     "category": "single_hop", "gt_article_ids": ["acct-008"],
     "reference_answer": "On sign-in from a new device, and on sensitive changes such as email or payment method. Already signed-in devices are unaffected.",
     "notes": ""},

    # --- devices ---
    {"question_id": "sh-043",
     "question": "Does StreamFlix work on a 2016 Samsung TV?",
     "category": "single_hop", "gt_article_ids": ["dev-001"],
     "reference_answer": "No. Samsung support starts with 2018 models; smart TVs from before 2018 are unsupported. A streaming stick is the usual workaround.",
     "notes": "Negative answer requiring a specific year cutoff."},

    {"question_id": "sh-044",
     "question": "Can I watch on my PlayStation?",
     "category": "single_hop", "gt_article_ids": ["dev-001"],
     "reference_answer": "Yes — PlayStation 4 and PlayStation 5 are supported.",
     "notes": ""},

    {"question_id": "sh-045",
     "question": "Typing my password with the TV remote is painful. Is there another way?",
     "category": "single_hop", "gt_article_ids": ["dev-002"],
     "reference_answer": "Yes — most TVs support code-based sign-in. The app shows a code you enter at streamflix.example/tv on a phone or computer.",
     "notes": ""},

    {"question_id": "sh-046",
     "question": "What Roku software version do I need?",
     "category": "single_hop", "gt_article_ids": ["dev-003"],
     "reference_answer": "Roku OS 9.4 or later, updated via Settings → System → System update.",
     "notes": "Near-duplicate cluster: must return dev-003, not dev-004/005."},

    {"question_id": "sh-047",
     "question": "StreamFlix keeps crashing on my Roku. How do I properly reset it?",
     "category": "single_hop", "gt_article_ids": ["dev-003"],
     "reference_answer": "Remove the channel, restart the Roku from Settings → System → Power → System restart, then re-add the channel. This clears cached data a simple restart leaves in place.",
     "notes": "Near-duplicate cluster."},

    {"question_id": "sh-048",
     "question": "Which Apple TV models can I use?",
     "category": "single_hop", "gt_article_ids": ["dev-004"],
     "reference_answer": "Apple TV 4th generation or later running tvOS 15 or later. Older models are unsupported.",
     "notes": "Near-duplicate cluster."},

    {"question_id": "sh-049",
     "question": "How do I clear the app cache on Fire TV?",
     "category": "single_hop", "gt_article_ids": ["dev-005"],
     "reference_answer": "Settings → Applications → Manage Installed Applications → StreamFlix → Clear cache. Clear data is the second step and signs you out.",
     "notes": "Near-duplicate cluster."},

    {"question_id": "sh-050",
     "question": "Does the cheap Fire TV Stick do 4K?",
     "category": "single_hop", "gt_article_ids": ["dev-005"],
     "reference_answer": "No. 4K needs Fire TV Stick 4K, 4K Max, or Fire TV Cube plus a Premium plan; the basic Fire TV Stick outputs 1080p maximum.",
     "notes": "Near-duplicate cluster, negative answer."},

    {"question_id": "sh-051",
     "question": "Can I stop the app using mobile data?",
     "category": "single_hop", "gt_article_ids": ["dev-006"],
     "reference_answer": "Yes — App settings → Video playback lets you restrict streaming to Wi-Fi or cap cellular quality. The default is Automatic, which uses less data on cellular.",
     "notes": ""},

    {"question_id": "sh-052",
     "question": "Why is Chrome capped at 1080p when I pay for Premium?",
     "category": "single_hop", "gt_article_ids": ["dev-007"],
     "reference_answer": "Chrome and Firefox are limited to 1080p by the digital-rights restrictions those browsers enforce, not by StreamFlix. Edge and Safari support 4K on compatible systems.",
     "notes": ""},

    {"question_id": "sh-053",
     "question": "Playback fails in my browser but works on my phone.",
     "category": "single_hop", "gt_article_ids": ["dev-007"],
     "reference_answer": "Clear the browser cache, disable extensions — ad blockers and privacy extensions are the most common cause — and confirm the OS and browser are current.",
     "notes": ""},

    {"question_id": "sh-054",
     "question": "Casting from my phone keeps dropping.",
     "category": "single_hop", "gt_article_ids": ["dev-008"],
     "reference_answer": "Confirm both devices are on the same network (guest networks and split 2.4/5 GHz SSIDs are frequent causes), restart the receiving device, update the app, and disable any VPN on the phone.",
     "notes": ""},

    {"question_id": "sh-055",
     "question": "If I cast to a 4K TV from my phone, do I get 4K?",
     "category": "single_hop", "gt_article_ids": ["dev-008"],
     "reference_answer": "Only if your plan allows it. Resolution follows the plan and receiving device, not the phone — a Basic plan casts at 720p even to a 4K TV.",
     "notes": ""},

    {"question_id": "sh-056",
     "question": "Can I still use StreamFlix on Xbox 360?",
     "category": "single_hop", "gt_article_ids": ["dev-009"],
     "reference_answer": "No. PlayStation 3 and Xbox 360 apps were retired in 2024 and will not install or sign in.",
     "notes": "Negative answer."},

    # --- content ---
    {"question_id": "sh-057",
     "question": "A movie I saved has disappeared. Why?",
     "category": "single_hop", "gt_article_ids": ["cont-001"],
     "reference_answer": "Licensed titles leave when their term ends and is not renewed. A title removed from the catalogue also leaves My List. Expiring titles show a 'Last day to watch' badge and appear in 'Leaving soon' for 30 days beforehand.",
     "notes": ""},

    {"question_id": "sh-058",
     "question": "Do StreamFlix Originals ever leave?",
     "category": "single_hop", "gt_article_ids": ["cont-001"],
     "reference_answer": "No. Originals are owned outright and do not expire.",
     "notes": ""},

    {"question_id": "sh-059",
     "question": "Will I see the same shows when I'm on holiday abroad?",
     "category": "single_hop", "gt_article_ids": ["cont-002"],
     "reference_answer": "No — the catalogue follows your current location, not your billing address, so you see the local catalogue. Originals are available everywhere StreamFlix operates. Downloads made before travelling remain playable.",
     "notes": ""},

    {"question_id": "sh-060",
     "question": "How many titles fit in My List?",
     "category": "single_hop", "gt_article_ids": ["cont-005"],
     "reference_answer": "Up to 500 titles per profile.",
     "notes": ""},

    # =================================================================
    # MULTI-HOP  (20)
    # =================================================================
    {"question_id": "mh-001",
     "question": "I'm on Basic — can two people watch different shows at once, and what quality do they get?",
     "category": "multi_hop", "gt_article_ids": ["strm-007", "strm-001"],
     "reference_answer": "No. Basic allows only 1 simultaneous stream, and its maximum resolution is 720p. Standard would allow 2 streams at up to 1080p.",
     "notes": "Requires the stream table and the resolution table."},

    {"question_id": "mh-002",
     "question": "I want 4K on my Roku — what do I need overall?",
     "category": "multi_hop", "gt_article_ids": ["dev-003", "strm-001", "strm-003"],
     "reference_answer": "A 4K-capable Roku model with HDMI 2.0, a Premium plan (Basic and Standard cap below 4K), and sustained bandwidth of 15 Mbps minimum / 25 Mbps recommended.",
     "notes": "Device + plan + bandwidth."},

    {"question_id": "mh-003",
     "question": "My trial is ending and my card just expired. What's going to happen?",
     "category": "multi_hop", "gt_article_ids": ["trial-002", "bill-004"],
     "reference_answer": "The subscription converts on day 15 and the charge fails. StreamFlix then retries up to four times over eight days, with access continuing during that window, and suspends access on day 9 if all retries fail. Adding a valid payment method restores access immediately.",
     "notes": "Trial conversion + failed-payment retry schedule."},

    {"question_id": "mh-004",
     "question": "I cancelled two months ago and want to come back — will my profiles still be there and will I get another free trial?",
     "category": "multi_hop", "gt_article_ids": ["bill-003", "trial-003"],
     "reference_answer": "Profiles and viewing history are retained for 10 months after cancellation, so resubscribing within that window restores them. The free trial is not offered again — it is once per household and returning subscribers are ineligible.",
     "notes": "History retention + trial eligibility."},

    {"question_id": "mh-005",
     "question": "How many devices can download, and how long do those downloads last?",
     "category": "multi_hop", "gt_article_ids": ["strm-006"],
     "reference_answer": "Device limits follow the plan: Basic 1, Standard 2, Premium 4. Downloads generally expire 30 days after download or 48 hours after playback begins, whichever is first.",
     "notes": "Both facts in one article — tests whether both are extracted."},

    {"question_id": "mh-006",
     "question": "Someone got into my account. How do I lock it down and can I get the charges back?",
     "category": "multi_hop", "gt_article_ids": ["acct-007", "bill-002"],
     "reference_answer": "Change the password, sign out of all devices, review billing, delete unfamiliar profiles, and enable 2FA. Unauthorised charges on a compromised account are refund-eligible; request via Help → Contact us.",
     "notes": "Security response + refund eligibility."},

    {"question_id": "mh-007",
     "question": "I'm on Standard and want HDR. What do I have to change?",
     "category": "multi_hop", "gt_article_ids": ["strm-008", "bill-007"],
     "reference_answer": "Upgrade to Premium — HDR is Premium-only. Upgrades take effect immediately with a prorated charge for the current cycle. You also need an HDR-capable display and a title mastered in HDR.",
     "notes": "Feature gate + upgrade mechanics."},

    {"question_id": "mh-008",
     "question": "Will deleting my account also stop the billing?",
     "category": "multi_hop", "gt_article_ids": ["acct-006", "bill-003"],
     "reference_answer": "Cancel the subscription first — deleting does not automatically stop billing purchased through a third party such as an app store, which must be cancelled with that provider. Cancelling keeps access to period end; deleting is permanent and removes all data.",
     "notes": "Cancel vs delete distinction."},

    {"question_id": "mh-009",
     "question": "Kids profile on the TV — how do I set one up and stop them raising the rating?",
     "category": "multi_hop", "gt_article_ids": ["acct-003", "acct-004"],
     "reference_answer": "Create it from Add profile with the Kids toggle enabled, set the maximum maturity rating in Manage profiles → Maturity settings, and set an account PIN in Account → Parental controls so changes require it.",
     "notes": ""},

    {"question_id": "mh-010",
     "question": "Two of us want to stream in 1080p at the same time — what plan and what internet speed?",
     "category": "multi_hop", "gt_article_ids": ["strm-007", "strm-001", "strm-003"],
     "reference_answer": "Standard or Premium (Basic allows 1 stream and caps at 720p). Two 1080p streams need roughly 16 Mbps sustained — 8 Mbps recommended per stream.",
     "notes": "Plan + resolution + additive bandwidth."},

    {"question_id": "mh-011",
     "question": "Charged after I cancelled — what do I do?",
     "category": "multi_hop", "gt_article_ids": ["bill-003", "bill-002"],
     "reference_answer": "A charge after a confirmed cancellation is refund-eligible. Note the two articles state different windows — bill-003 says 14 days from the charge, bill-002 says 30 days — so contact support with the charge date and amount.",
     "notes": "CONTRADICTION PROBE — a good answer surfaces the conflict."},

    {"question_id": "mh-012",
     "question": "Does a cast stream count towards my limit, and what quality will it be?",
     "category": "multi_hop", "gt_article_ids": ["dev-008", "strm-007"],
     "reference_answer": "Yes — a cast stream counts as one simultaneous stream. Quality follows the plan and receiving device, not the phone.",
     "notes": ""},

    {"question_id": "mh-013",
     "question": "I'm travelling next week — how do I make sure I can watch my shows?",
     "category": "multi_hop", "gt_article_ids": ["cont-002", "strm-006"],
     "reference_answer": "The catalogue follows your current location, so titles may be unavailable abroad. Download before travelling on the mobile apps — downloads remain playable while their licence lasts. Device limits follow the plan and downloads expire 30 days after download or 48 hours after playback starts.",
     "notes": ""},

    {"question_id": "mh-014",
     "question": "Why can't I find a show my friend in another country is watching?",
     "category": "multi_hop", "gt_article_ids": ["cont-002", "cont-004"],
     "reference_answer": "Catalogues are licensed region by region, so a title available in one country may be absent in another. Search results are limited to titles available in your current region. Originals are the exception and available everywhere.",
     "notes": ""},

    {"question_id": "mh-015",
     "question": "What's the total cost difference between Basic and Premium, and what do I get for it?",
     "category": "multi_hop", "gt_article_ids": ["bill-001", "strm-001", "strm-007", "strm-006"],
     "reference_answer": "Basic $9 vs Premium $19 — $10 more per month. Premium adds 4 simultaneous streams (vs 1), 4K UHD (vs 720p), 4 download devices (vs 1), plus HDR and Dolby Atmos eligibility.",
     "notes": "Four-article synthesis — hardest retrieval case in the set."},

    {"question_id": "mh-016",
     "question": "My payment failed and I'm worried I'll lose my watchlist.",
     "category": "multi_hop", "gt_article_ids": ["bill-004", "cont-005"],
     "reference_answer": "Profiles, viewing history, and My List are preserved during suspension. Updating a valid payment method restores access immediately.",
     "notes": ""},

    {"question_id": "mh-017",
     "question": "I changed my mind during the trial. Will I definitely not be charged?",
     "category": "multi_hop", "gt_article_ids": ["trial-004", "trial-002"],
     "reference_answer": "Cancel before day 15 and no charge is made; access continues to the end of the 14-day period. Confirm the cancellation email arrived — without it, the cancellation did not complete and the day-15 charge will proceed.",
     "notes": ""},

    {"question_id": "mh-018",
     "question": "Subtitles are missing in the language I want — is that my settings or the title?",
     "category": "multi_hop", "gt_article_ids": ["strm-005", "cont-002"],
     "reference_answer": "Subtitle availability is set per title and per region by the licensor, so a language present in one country may be absent in another for the same title. Styling and track selection are in the player and Account → Subtitle appearance.",
     "notes": ""},

    {"question_id": "mh-019",
     "question": "How do I get better recommendations for my partner and me — we like completely different things?",
     "category": "multi_hop", "gt_article_ids": ["cont-004", "acct-003"],
     "reference_answer": "Use separate profiles — recommendations are built per profile and a shared profile produces suggestions that suit nobody. Up to 5 profiles are available on every plan. Rating titles and hiding unwanted ones from viewing activity also improves suggestions.",
     "notes": ""},

    {"question_id": "mh-020",
     "question": "New show comes out next week — how do I get told, and will I get it at the same time as the US?",
     "category": "multi_hop", "gt_article_ids": ["cont-003"],
     "reference_answer": "Enable release notifications on the title's details page (OS-level notifications must also be on), or use Remind me in the Coming soon row. Originals release simultaneously worldwide; licensed titles sometimes have staggered release dates by region.",
     "notes": ""},

    # =================================================================
    # AMBIGUOUS  (15)
    # =================================================================
    {"question_id": "am-001",
     "question": "How do I get a refund?",
     "category": "ambiguous", "gt_article_ids": ["bill-002", "bill-003"],
     "reference_answer": "Refunds are limited to specific situations (duplicate charge, charge after confirmed cancellation, outage over 24 hours, unauthorised charge). Request via Account → Help → Contact us with the charge date and amount. Note the corpus states two different windows — 30 days in the refund policy and 14 days in the cancellation article.",
     "notes": "CONTRADICTION PROBE — both articles are legitimately relevant."},

    {"question_id": "am-002",
     "question": "How do I set up StreamFlix on my streaming box?",
     "category": "ambiguous", "gt_article_ids": ["dev-003", "dev-004", "dev-005", "dev-002"],
     "reference_answer": "The steps depend on the device. Broadly: open the device's app store, search for StreamFlix, install, then sign in — most support an activation code entered at streamflix.example/tv. Ask which device to give exact steps.",
     "notes": "Near-duplicate cluster — tests whether the system asks for clarification."},

    {"question_id": "am-003",
     "question": "What plan should I get?",
     "category": "ambiguous", "gt_article_ids": ["bill-001", "strm-007", "strm-001"],
     "reference_answer": "Depends on simultaneous viewers and desired quality: Basic $9 (1 stream, 720p), Standard $14 (2 streams, 1080p), Premium $19 (4 streams, 4K plus HDR/Atmos eligibility).",
     "notes": ""},

    {"question_id": "am-004",
     "question": "It's not working.",
     "category": "ambiguous", "gt_article_ids": ["strm-002", "dev-007", "bill-004"],
     "reference_answer": "Too vague to answer directly — a good response asks what specifically fails (playback, sign-in, billing) and on which device.",
     "notes": "Extreme vagueness — should trigger clarification, not a guess."},

    {"question_id": "am-005",
     "question": "How do I change my settings?",
     "category": "ambiguous", "gt_article_ids": ["strm-001", "acct-003", "strm-005"],
     "reference_answer": "Depends which settings: playback quality (Account → Playback settings), profiles (Manage profiles), subtitle appearance (Account → Subtitle appearance), or parental controls (Account → Parental controls).",
     "notes": ""},

    {"question_id": "am-006",
     "question": "Why can't I watch this?",
     "category": "ambiguous", "gt_article_ids": ["cont-002", "cont-001", "acct-004", "strm-007"],
     "reference_answer": "Several possible causes: the title is not licensed in your region, it has left the catalogue, a maturity setting blocks it on this profile, or the simultaneous-stream limit is reached.",
     "notes": ""},

    {"question_id": "am-007",
     "question": "Can I share my account?",
     "category": "ambiguous", "gt_article_ids": ["strm-007", "acct-003"],
     "reference_answer": "The corpus documents profile counts (up to 5) and simultaneous-stream limits by plan, but does not state a household or password-sharing policy. A correct answer covers what is documented and flags that sharing rules are not covered.",
     "notes": "PARTIAL COVERAGE — adjacent facts exist, the actual policy does not."},

    {"question_id": "am-008",
     "question": "What's included in my subscription?",
     "category": "ambiguous", "gt_article_ids": ["bill-001", "strm-007", "strm-006", "acct-003"],
     "reference_answer": "Full catalogue access for your region; simultaneous streams, maximum resolution, and download devices by plan tier; and up to 5 profiles on every plan.",
     "notes": ""},

    {"question_id": "am-009",
     "question": "How do I cancel?",
     "category": "ambiguous", "gt_article_ids": ["bill-003", "trial-004"],
     "reference_answer": "Account → Membership → Cancel membership, then confirm; a confirmation email follows. During a trial, cancelling before day 15 avoids any charge. On a paid plan, access continues to the end of the current billing period.",
     "notes": "Trial vs paid path."},

    {"question_id": "am-010",
     "question": "The quality is bad.",
     "category": "ambiguous", "gt_article_ids": ["strm-001", "strm-002", "strm-003", "dev-007"],
     "reference_answer": "Could be plan resolution cap, bandwidth, a playback-settings cap, or a browser 1080p limit. A good response narrows down device and plan before advising.",
     "notes": ""},

    {"question_id": "am-011",
     "question": "How do I contact a human?",
     "category": "ambiguous", "gt_article_ids": ["bill-002", "acct-007"],
     "reference_answer": "The corpus references Account → Help → Contact us for refunds and unrecognised charges, but does not document support hours, channels, or response times. A correct answer gives the path and flags the rest as uncovered.",
     "notes": "PARTIAL COVERAGE."},

    {"question_id": "am-012",
     "question": "Why is my account locked?",
     "category": "ambiguous", "gt_article_ids": ["bill-004", "acct-007"],
     "reference_answer": "Most likely a failed payment — access suspends on day 9 after four failed retries. Alternatively, unauthorised access may have triggered a password reset, which signs out all devices.",
     "notes": ""},

    {"question_id": "am-013",
     "question": "Do I get a discount?",
     "category": "ambiguous", "gt_article_ids": ["bill-008", "trial-001"],
     "reference_answer": "Promotional rates apply for a fixed number of cycles when a valid code or link is used, and only one can be active at a time. New subscribers get a 14-day free trial, subject to once-per-household eligibility.",
     "notes": ""},

    {"question_id": "am-014",
     "question": "How many people can use my account?",
     "category": "ambiguous", "gt_article_ids": ["acct-003", "strm-007"],
     "reference_answer": "Up to 5 profiles on any plan, but simultaneous streaming is capped by tier: Basic 1, Standard 2, Premium 4. Profiles and streams are separate limits.",
     "notes": ""},

    {"question_id": "am-015",
     "question": "What's the best quality I can get?",
     "category": "ambiguous", "gt_article_ids": ["strm-001", "strm-008", "dev-007", "dev-001"],
     "reference_answer": "4K UHD on Premium, plus HDR and Dolby Atmos on eligible titles. Actual maximum also depends on the device and, in browsers, on the browser — Chrome and Firefox cap at 1080p.",
     "notes": ""},

    # =================================================================
    # OUT-OF-SCOPE  (25)  — must be refused
    # =================================================================
    {"question_id": "oos-001", "question": "How do I buy a StreamFlix gift card?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered by the help centre. The assistant should say so and offer to route to support.",
     "notes": "GAP: gift subscriptions."},

    {"question_id": "oos-002", "question": "Can I gift a subscription to my mum for her birthday?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: gift subscriptions."},

    {"question_id": "oos-003", "question": "Do you offer accounts for businesses?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: business accounts."},

    {"question_id": "oos-004", "question": "We want StreamFlix for our school — is there an education plan?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: education accounts."},

    {"question_id": "oos-005", "question": "Can I show StreamFlix in my café?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: commercial licensing."},

    {"question_id": "oos-006", "question": "Does StreamFlix work with a screen reader?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: accessibility."},

    {"question_id": "oos-007", "question": "Is there audio description for blind viewers?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered. Note the subtitles article is adjacent but does not cover audio description.",
     "notes": "GAP: accessibility — adjacent article makes this a hard refusal."},

    {"question_id": "oos-008", "question": "Can I watch live football on StreamFlix?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: live sports."},

    {"question_id": "oos-009", "question": "Do you stream live events or concerts?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: live events."},

    {"question_id": "oos-010", "question": "Am I allowed to give my password to my brother in another city?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered — no password-sharing policy is documented.",
     "notes": "GAP: password sharing. Stream-limit articles are adjacent but not the policy."},

    {"question_id": "oos-011", "question": "How much is the extra member add-on?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: extra-member fees."},

    {"question_id": "oos-012", "question": "I want a copy of all the data you hold on me.",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered — the privacy article covers viewing history and retention but not subject access requests.",
     "notes": "GAP: GDPR data export."},

    {"question_id": "oos-013", "question": "How do I file a GDPR request?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: GDPR."},

    {"question_id": "oos-014", "question": "Is there an affiliate programme I can join?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: affiliate programme."},

    {"question_id": "oos-015", "question": "Where can I buy StreamFlix merchandise?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: merchandise."},

    {"question_id": "oos-016", "question": "Do you sell DVDs of your Originals?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: physical media."},

    {"question_id": "oos-017", "question": "How do I pitch my screenplay to StreamFlix?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: content submissions."},

    {"question_id": "oos-018", "question": "Are you hiring engineers?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: careers."},

    {"question_id": "oos-019", "question": "What was StreamFlix's revenue last quarter?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: corporate financials."},

    {"question_id": "oos-020", "question": "Can I invest in StreamFlix?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered.", "notes": "GAP: investor relations."},

    {"question_id": "oos-021", "question": "What's the weather in Seattle tomorrow?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Entirely unrelated to StreamFlix support.",
     "notes": "GAP: off-domain — trivially out of scope, sanity floor."},

    {"question_id": "oos-022", "question": "Write me a Python script to scrape your catalogue.",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not a support question and should not be assisted.",
     "notes": "GAP: off-task request."},

    {"question_id": "oos-023", "question": "Which is better value, StreamFlix or Netflix?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered — no competitor comparison exists in the help centre.",
     "notes": "GAP: competitor comparison, invites speculation."},

    {"question_id": "oos-024", "question": "When is season 3 of my favourite Original coming out?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered — the releases article explains where to find dates but the corpus holds no title-specific schedule.",
     "notes": "GAP: title-specific data. Adjacent article makes this a strong hallucination probe."},

    {"question_id": "oos-025", "question": "Can I get a discount if I complain about the buffering?",
     "category": "out_of_scope", "gt_article_ids": [],
     "reference_answer": "Not covered — refunds require an outage over 24 continuous hours; no goodwill-credit policy is documented.",
     "notes": "GAP: goodwill credits. Refund article is adjacent but does not authorise this."},
]


# ---------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------
def by_category(category: str) -> list[dict]:
    """All questions in one category."""
    return [q for q in GOLDEN_QUESTIONS if q["category"] == category]


def category_counts() -> dict[str, int]:
    """Question count per category."""
    counts: dict[str, int] = {}
    for q in GOLDEN_QUESTIONS:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    return counts


def articles_referenced() -> set[str]:
    """Every article id cited as ground truth by at least one question."""
    return {aid for q in GOLDEN_QUESTIONS for aid in q["gt_article_ids"]}
