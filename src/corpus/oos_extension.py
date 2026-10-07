"""275 more questions the corpus cannot answer.

The golden set holds 25 out-of-scope questions, and the decision memo's
recommendation hangs on them: the cited arm refused 25 of 25, and 25 of
25 only bounds the true refusal rate at 86.7%. The memo named the fix —
"roughly 300 clean out-of-scope questions" — and this is that set. With
the original 25 it makes 300.

It is kept OUT of the golden set on purpose. The golden set was written
before any retrieval code existed, and every number in Phases 1 to 7 is
computed on it; adding 275 questions afterwards would change all of
them and blur the one property that makes the golden set worth
trusting. This is a follow-up sample for one question, how often does
the system answer when it should refuse, and it is reported separately.

What it is not. These questions were written AFTER the 25-question
result was known, by the same author as the corpus. They are not a
blind sample of what customers ask. They widen the sample; they do not
make it independent.

Three tiers, hardest first:

  near_miss   the corpus has an adjacent article that is silent on the
              fact asked for. Retrieval returns something plausible and
              the generator has material to improvise from. This is the
              tier where a support bot gets someone hurt.
  gap         a whole topic the help centre does not cover.
  off_domain  nothing to do with the product, including questions about
              OTHER streaming services (where answering with StreamFlix's
              own procedure is the trap) and general knowledge the model
              knows and the context does not contain.

Two guards against the set's own errors:

  - Every question names at least one term that must NOT appear anywhere
    in the corpus. A test enforces it. It cannot prove a question is
    unanswerable, but it catches the commonest way to get this wrong:
    writing an "unanswerable" question about something an article
    actually mentions.
  - Questions are deliberately NOT of the form "is X on your list?".
    Where the corpus gives a complete-looking list (accepted cards,
    supported consoles), "X is not listed" is a fair answer drawn from
    the context, and counting it as a failure would be scoring the
    system down for being right.

Each question also carries a `topic`. Questions on one topic are not
independent draws, so results are reported per topic as well as per
question, and no topic has more than three.
"""
from __future__ import annotations

# (topic, tier, question, terms that must be absent from the corpus)
_QUESTIONS: list[tuple[str, str, str, tuple[str, ...]]] = [
    # ------------------------------------------------------------------
    # near_miss — pricing and discounts
    # ------------------------------------------------------------------
    ("student discount", "near_miss", "Is there a student discount?", ("student",)),
    ("military discount", "near_miss", "Do military veterans get a reduced rate?", ("military", "veteran")),
    ("senior discount", "near_miss", "Is there a discount for pensioners or over-65s?", ("pensioner", "senior")),
    ("teacher discount", "near_miss", "Do teachers get money off?", ("teacher",)),
    ("healthcare discount", "near_miss", "Is there a discount for nurses and other healthcare workers?", ("nurse", "healthcare")),
    ("hardship rate", "near_miss", "I've lost my job. Do you have a hardship or low-income rate?", ("hardship", "low-income")),
    ("charity discount", "near_miss", "Do registered charities get a discounted subscription?", ("charity", "charities")),
    ("annual billing", "near_miss", "Can I pay yearly instead of monthly, and is it cheaper?", ("yearly", "annual")),
    ("prepay discount", "near_miss", "Do I get a discount for paying six months upfront?", ("upfront",)),
    ("membership freeze", "near_miss", "Can I freeze my membership for two months without cancelling?", ("freeze",)),
    ("family plan", "near_miss", "Is there a family plan that covers two separate homes on one bill?", ("family plan",)),
    ("ad-supported tier", "near_miss", "Do you have a cheaper plan with adverts?", ("ad-supported", "commercials")),
    ("mobile-only plan", "near_miss", "Is there a cheaper plan that only works on phones?", ("mobile-only",)),
    ("lifetime subscription", "near_miss", "Can I buy a lifetime subscription?", ("lifetime",)),
    ("day pass", "near_miss", "Can I buy a day pass or a weekend pass instead of a month?", ("day pass", "weekend pass")),
    ("rentals", "near_miss", "Can I rent a single film without subscribing?", ("rent", "rental")),
    ("price match", "near_miss", "Will you price-match a competitor's offer?", ("price-match", "competitor")),
    ("seasonal deals", "near_miss", "Is there a Black Friday deal this year?", ("black friday",)),
    ("price guarantee", "near_miss", "Can I lock in today's price for two years?", ("guarantee", "lock in")),
    ("future price changes", "near_miss", "When is the next price rise planned?", ("price rise", "price increase")),
    ("loyalty rewards", "near_miss", "Do long-time subscribers get a loyalty reward?", ("loyalty", "reward")),
    ("referrals", "near_miss", "Do I get a bonus if I refer a friend?", ("referral", "refer")),
    ("cashback", "near_miss", "Can I earn cashback or air miles paying for StreamFlix?", ("cashback", "miles")),
    ("late fees", "near_miss", "What's the late fee if my payment fails?", ("late fee",)),
    ("reactivation fee", "near_miss", "Is there a reactivation fee after my account is suspended?", ("reactivation",)),
    ("split payment", "near_miss", "Can I split the monthly charge across two cards?", ("split",)),
    ("paper receipts", "near_miss", "Can you post me a paper receipt every month?", ("paper", "postal")),
    ("business invoicing", "near_miss", "Can the invoice show my company name so I can claim it as an expense?", ("company name", "expense")),
    ("tax identifiers", "near_miss", "What's your VAT registration number?", ("registration number",)),
    ("carrier bundles", "near_miss", "Is StreamFlix included with my Vodafone phone contract?", ("vodafone",)),
    ("money-back guarantee", "near_miss", "Do you offer a money-back guarantee if I don't like it?", ("money-back",)),
    ("store credit", "near_miss", "Can I take my refund as account credit instead of back to my card?", ("account credit", "store credit")),
    ("chargebacks", "near_miss", "What happens to my account if I do a chargeback through my bank?", ("chargeback",)),
    ("credit reporting", "near_miss", "Will a failed payment affect my credit score?", ("credit score", "credit bureau")),
    ("debt collection", "near_miss", "Do you pass unpaid bills to a debt collection agency?", ("debt", "collection agency")),
    ("cancel by phone", "near_miss", "Can I cancel by phone instead of online?", ("by phone", "telephone")),
    ("cooling-off rights", "near_miss", "Is there a statutory cooling-off period after I sign up?", ("cooling-off", "statutory")),
    ("retention offers", "near_miss", "If I say I'm leaving, will you offer me a retention deal to stay?", ("retention deal",)),
    ("premium channels", "near_miss", "Can I add a premium channel like HBO to my plan?", ("hbo",)),
    ("sports add-on", "near_miss", "How much is the sports add-on?", ("add-on", "sports")),
    ("branded card", "near_miss", "Is there a StreamFlix-branded credit card with rewards?", ("branded",)),
    ("company failure", "near_miss", "What happens to my subscription if StreamFlix goes bankrupt?", ("bankrupt",)),
    # ------------------------------------------------------------------
    # near_miss — account and privacy
    # ------------------------------------------------------------------
    ("minimum age", "near_miss", "What's the minimum age to open an account?", ("minimum age",)),
    ("name change", "near_miss", "I got married — how do I change the legal name on my account?", ("married", "legal name")),
    ("account ownership", "near_miss", "Can I hand ownership of my account over to my partner?", ("ownership",)),
    ("profile migration", "near_miss", "Can I migrate my profile to a different account and keep my history?", ("migrate",)),
    ("bereavement", "near_miss", "My father died. Can the family keep using his account?", ("died", "deceased", "bereavement")),
    ("social login", "near_miss", "Can I sign in with my Facebook login?", ("facebook", "social login")),
    ("passkeys", "near_miss", "Do you support passkeys instead of a password?", ("passkey", "passkeys")),
    ("hardware keys", "near_miss", "Can I use a YubiKey as my second factor?", ("yubikey", "security key")),
    ("biometrics", "near_miss", "Can I unlock the app with my fingerprint or Face ID?", ("fingerprint", "face id", "biometric")),
    ("lockout policy", "near_miss", "How many wrong passwords before I'm locked out?", ("locked out", "lockout")),
    ("security questions", "near_miss", "Which security questions can I set up?", ("security question", "security questions")),
    ("username", "near_miss", "How do I change my username?", ("username",)),
    ("joint owners", "near_miss", "Can I add my wife as a joint account holder?", ("joint", "account holder")),
    ("identity documents", "near_miss", "For identity verification, do you accept a passport or a driving licence?", ("passport", "driving licence")),
    ("customer number", "near_miss", "Where do I find my customer ID number?", ("customer id", "account number")),
    ("proof of subscription", "near_miss", "Can you send a letter confirming my subscription as proof of address?", ("letter", "proof of address")),
    ("history export", "near_miss", "Can I export my viewing history as a spreadsheet?", ("export", "spreadsheet")),
    ("yearly stats", "near_miss", "Is there a year-in-review showing how many hours I watched?", ("year-in-review", "statistics")),
    ("data selling", "near_miss", "Do you sell my viewing data to advertisers?", ("sell", "advertisers")),
    ("law enforcement", "near_miss", "Do you hand viewing records to the police?", ("police", "law enforcement")),
    ("data location", "near_miss", "Which country are your servers in?", ("servers", "data centre")),
    ("encryption", "near_miss", "Is my personal data encrypted?", ("encrypted", "encryption")),
    ("tracking opt-out", "near_miss", "How do I opt out of ad tracking?", ("opt out",)),
    ("marketing emails", "near_miss", "How do I unsubscribe from your marketing emails?", ("unsubscribe", "marketing")),
    ("account bans", "near_miss", "What can get my account permanently banned?", ("banned", "ban")),
    ("appeals", "near_miss", "How do I appeal if you close my account for breaking the rules?", ("appeal",)),
    ("profile name rules", "near_miss", "What words are prohibited in profile names?", ("prohibited", "offensive")),
    ("custom avatars", "near_miss", "Can I upload my own photo as a profile picture?", ("upload", "photo")),
    ("account resale", "near_miss", "Am I allowed to sell my account to someone else?", ("sell", "resale")),
    ("guest passes", "near_miss", "Can I give a friend a guest pass for a week?", ("guest pass",)),
    ("beta features", "near_miss", "How do I get early access to beta features?", ("beta",)),
    # ------------------------------------------------------------------
    # near_miss — parental controls
    # ------------------------------------------------------------------
    ("screen time limits", "near_miss", "Can I set a daily screen time limit on my child's profile?", ("screen time", "time limit")),
    ("viewing reports", "near_miss", "Can you email me a weekly report of what my kids watched?", ("weekly report",)),
    ("title approval", "near_miss", "Can I approve each title before my child is allowed to watch it?", ("approve", "approval")),
    ("forgotten PIN", "near_miss", "I forgot the parental PIN. How do I reset the PIN?", ("reset the pin", "forgot the pin")),
    ("age verification", "near_miss", "Do you check ID to verify age for adult titles?", ("age verification", "adult")),
    # ------------------------------------------------------------------
    # near_miss — content and catalogue
    # ------------------------------------------------------------------
    ("specific title", "near_miss", "When does the final season of Harbour Lights arrive?", ("harbour lights",)),
    ("specific title", "near_miss", "How many episodes are in season two of Northline?", ("northline",)),
    ("cast details", "near_miss", "Who is starring in your new Originals thriller?", ("starring", "thriller")),
    ("runtimes", "near_miss", "What's the runtime of the extended director's cut?", ("runtime", "director's cut")),
    ("awards", "near_miss", "Which of your Originals have won awards?", ("award", "awards")),
    ("catalogue size", "near_miss", "How many films are in the catalogue altogether?", ("altogether", "thousand")),
    ("genres", "near_miss", "Do you have anime?", ("anime",)),
    ("studios", "near_miss", "Which film studios' back catalogues do you carry?", ("studio", "studios")),
    ("popularity charts", "near_miss", "What's the most-watched show on StreamFlix this week?", ("most-watched", "trending")),
    ("trailers", "near_miss", "Can I watch trailers for films that aren't out yet?", ("trailer", "trailers")),
    ("bonus material", "near_miss", "Do you include deleted scenes or behind-the-scenes extras?", ("deleted scenes", "behind-the-scenes")),
    ("viewer comments", "near_miss", "Can I write a comment under a show for other viewers to read?", ("comment",)),
    ("social features", "near_miss", "Can I see what my friends are watching?", ("friends",)),
    ("watch parties", "near_miss", "Is there a watch party feature for watching with people in other houses?", ("watch party",)),
    ("playlists", "near_miss", "Can I make a playlist and share it with another account?", ("playlist",)),
    ("games", "near_miss", "Does my plan include any mobile games?", ("mobile games",)),
    ("audio content", "near_miss", "Do you have podcasts or music?", ("podcast", "podcasts", "music")),
    ("interactive titles", "near_miss", "Do you have interactive choose-your-own-adventure titles?", ("interactive",)),
    ("renewal votes", "near_miss", "Can subscribers vote on which shows get another season?", ("vote",)),
    ("live TV", "near_miss", "Do you have live TV channels?", ("live tv",)),
    ("news", "near_miss", "Can I watch the news live?", ("news",)),
    ("adult content", "near_miss", "Is there an explicit 18+ section?", ("explicit", "18+")),
    ("censorship", "near_miss", "Do you censor films in some countries?", ("censor", "censored")),
    ("edited versions", "near_miss", "Are your films the uncut versions or edited for content?", ("uncut",)),
    ("product placement", "near_miss", "Do your Originals contain paid product placement?", ("product placement",)),
    ("sponsored rows", "near_miss", "Are any of the home-screen recommendations sponsored?", ("sponsored",)),
    ("country availability", "near_miss", "Is StreamFlix available in Japan?", ("japan",)),
    ("country availability", "near_miss", "Does StreamFlix work in China?", ("china",)),
    ("production languages", "near_miss", "Do you make Korean-language dramas?", ("korean",)),
    # ------------------------------------------------------------------
    # near_miss — player features
    # ------------------------------------------------------------------
    ("skip intro", "near_miss", "Can I make it skip the intro automatically?", ("intro",)),
    ("playback speed", "near_miss", "Can I play shows at 1.5x speed?", ("1.5x", "playback speed")),
    ("sleep timer", "near_miss", "Is there a sleep timer?", ("sleep timer",)),
    ("autoplay", "near_miss", "How do I turn off autoplay of the next episode?", ("autoplay",)),
    ("resolutions beyond 4K", "near_miss", "Which titles can I watch in 8K?", ("8k",)),
    ("3D", "near_miss", "Do you have 3D films?", ("3d",)),
    ("IMAX", "near_miss", "Do you support IMAX Enhanced?", ("imax",)),
    ("bitrate", "near_miss", "What bitrate are your 4K streams encoded at?", ("bitrate", "encoded")),
    ("codecs", "near_miss", "Do you stream in AV1?", ("av1",)),
    ("frame rate", "near_miss", "Do you support 120Hz high frame rate playback?", ("120hz", "frame rate")),
    ("aspect ratio", "near_miss", "Why are there black bars above and below some films?", ("black bars", "aspect ratio")),
    ("orientation", "near_miss", "Can I watch in portrait mode on my phone?", ("portrait",)),
    ("dark mode", "near_miss", "Does the app have a dark mode?", ("dark mode",)),
    ("themes", "near_miss", "Can I change the colour theme of the app?", ("theme",)),
    ("screenshots", "near_miss", "How do I take a screenshot of a scene?", ("screenshot",)),
    ("recording", "near_miss", "Can I record a show to keep permanently?", ("record",)),
    ("screen sharing", "near_miss", "Can I screen-share StreamFlix on a video call?", ("screen-share", "video call")),
    ("throttling", "near_miss", "Do you throttle streams in the evening?", ("throttle",)),
    ("zero-rating", "near_miss", "Is StreamFlix zero-rated on my mobile data plan?", ("zero-rated", "zero-rating")),
    ("audio-only mode", "near_miss", "Is there an audio-only mode to save data?", ("audio-only",)),
    ("transcripts", "near_miss", "Can I get a transcript of a film's dialogue?", ("transcript",)),
    ("dual subtitles", "near_miss", "Can I show two subtitle languages at once for language learning?", ("dual subtitles", "language learning")),
    ("custom subtitles", "near_miss", "Can I load my own subtitle file?", ("subtitle file",)),
    ("dubbing", "near_miss", "Who do I tell about a mistranslation in the dubbing?", ("dubbing", "dubbed", "mistranslation")),
    ("viewing caps", "near_miss", "Is there a cap on how many hours I can watch each month?", ("hours i can", "monthly cap")),
    ("download storage", "near_miss", "Can I save downloads to an SD card?", ("sd card", "memory card")),
    ("download transfer", "near_miss", "Can I copy my downloads to a laptop over USB?", ("usb",)),
    ("data warnings", "near_miss", "Can the app warn me when I've used 10 GB of mobile data?", ("warn", "warning")),
    # ------------------------------------------------------------------
    # near_miss — devices and home set-up
    # ------------------------------------------------------------------
    ("app size", "near_miss", "How many megabytes does the mobile app take up?", ("megabytes",)),
    ("in-car", "near_miss", "Does the app work with CarPlay or Android Auto?", ("carplay", "android auto")),
    ("in-flight", "near_miss", "Can I watch StreamFlix on a plane's seatback screen?", ("seatback", "plane")),
    ("wearables", "near_miss", "Is there a StreamFlix app for my smartwatch?", ("smartwatch",)),
    ("virtual reality", "near_miss", "Is there a VR app for the Meta Quest?", ("vr", "quest")),
    ("controllers", "near_miss", "Can I navigate with a Bluetooth game controller?", ("bluetooth", "controller")),
    ("keyboard and mouse", "near_miss", "Does the TV app work with a mouse?", ("mouse",)),
    ("Miracast", "near_miss", "Can I mirror my laptop to the TV with Miracast?", ("miracast",)),
    ("HDCP", "near_miss", "Why does my projector show an HDCP error?", ("hdcp", "projector")),
    ("sideloading", "near_miss", "Can I sideload the APK onto an unsupported Android box?", ("sideload", "apk")),
    ("StreamFlix hardware", "near_miss", "What's the warranty on the StreamFlix streaming stick?", ("warranty",)),
    ("picture calibration", "near_miss", "How should I calibrate my TV picture for StreamFlix?", ("calibrate", "calibration")),
    ("picture modes", "near_miss", "Should I turn motion smoothing off for StreamFlix?", ("motion smoothing",)),
    ("eARC", "near_miss", "Does StreamFlix pass Atmos through eARC to my soundbar?", ("earc", "passthrough")),
    ("headphones", "near_miss", "Can two pairs of Bluetooth headphones listen at once?", ("headphones", "bluetooth")),
    ("spatial audio", "near_miss", "Does the app support spatial audio on AirPods?", ("airpods", "spatial")),
    ("voice assistants", "near_miss", "Can I ask Alexa to play a show on StreamFlix?", ("alexa",)),
    ("voice control", "near_miss", "Can I control the TV app with voice commands?", ("voice commands", "voice control")),
    ("smart home", "near_miss", "Can I link StreamFlix to my smart home hub?", ("smart home",)),
    ("network settings", "near_miss", "Which firewall ports does StreamFlix need open?", ("firewall", "ports")),
    ("network settings", "near_miss", "What DNS settings should I use for StreamFlix?", ("dns",)),
    ("satellite internet", "near_miss", "Does StreamFlix work on Starlink?", ("starlink", "satellite")),
    ("at sea", "near_miss", "Will StreamFlix work on a cruise ship's Wi-Fi?", ("cruise", "ship")),
    ("CDN", "near_miss", "Which CDN do you use to deliver video?", ("cdn",)),
    ("uptime", "near_miss", "What uptime do you guarantee?", ("uptime", "sla")),
    # ------------------------------------------------------------------
    # gap — support channels
    # ------------------------------------------------------------------
    ("phone support", "gap", "What's your customer service phone number?", ("phone number", "telephone")),
    ("live chat", "gap", "What hours is live chat open?", ("live chat", "opening hours")),
    ("human agent", "gap", "How do I get through to a human agent?", ("agent", "human")),
    ("language support", "gap", "Do you have a support line in Spanish?", ("spanish",)),
    ("complaints", "gap", "What's the postal address for written complaints?", ("complaints", "postal")),
    ("complaints", "gap", "How do I escalate a complaint to an ombudsman?", ("ombudsman", "escalate")),
    ("response times", "gap", "What's your average response time to a support ticket?", ("response time", "ticket")),
    ("community forum", "gap", "Is there a community forum for subscribers?", ("forum",)),
    ("social support", "gap", "Do you answer support questions on Twitter?", ("twitter",)),
    ("callbacks", "gap", "Can I book a callback from support?", ("callback",)),
    ("out-of-hours", "gap", "Is support available 24/7?", ("24/7",)),
    # ------------------------------------------------------------------
    # gap — the company
    # ------------------------------------------------------------------
    ("leadership", "gap", "Who is the CEO of StreamFlix?", ("ceo",)),
    ("headquarters", "gap", "Where is StreamFlix headquartered?", ("headquartered", "headquarters")),
    ("history", "gap", "What year was StreamFlix founded?", ("founded",)),
    ("subscriber numbers", "gap", "How many million subscribers does StreamFlix have?", ("million",)),
    ("shares", "gap", "What's StreamFlix's stock ticker?", ("ticker", "stock")),
    ("ownership", "gap", "Which parent company owns StreamFlix?", ("parent company",)),
    ("mergers", "gap", "Will prices change after the merger?", ("merger",)),
    ("sustainability", "gap", "Is StreamFlix carbon neutral?", ("carbon",)),
    ("AI policy", "gap", "What's your policy on AI-generated content?", ("ai-generated", "artificial intelligence")),
    ("AI policy", "gap", "Do you use my viewing data to train AI models?", ("train", "ai models")),
    ("regulation", "gap", "Which regulator oversees StreamFlix — is it Ofcom?", ("ofcom", "regulator")),
    ("talent pay", "gap", "What residuals do you pay actors?", ("residuals", "actors")),
    ("sponsorship", "gap", "Do you sponsor film festivals?", ("festival", "sponsor")),
    ("content complaints", "gap", "How do I complain that a show is offensive?", ("offensive", "complain")),
    # ------------------------------------------------------------------
    # gap — legal
    # ------------------------------------------------------------------
    ("terms of service", "gap", "Do your terms of service force me into arbitration?", ("arbitration", "terms of service")),
    ("litigation", "gap", "Is there a class action against StreamFlix I can join?", ("class action",)),
    ("copyright", "gap", "How do I send you a DMCA takedown notice?", ("dmca", "takedown")),
    ("copyright", "gap", "How do I report copyright infringement on StreamFlix?", ("copyright", "infringement")),
    ("clip usage", "gap", "Can I use clips from your Originals in my YouTube video?", ("youtube", "clips")),
    ("data protection officer", "gap", "Who is your data protection officer?", ("data protection officer",)),
    ("privacy law", "gap", "Can I object to automated profiling under GDPR?", ("gdpr", "profiling")),
    ("privacy law", "gap", "Do you comply with the CCPA for California residents?", ("ccpa", "california")),
    ("security disclosure", "gap", "How do I report a security vulnerability I found?", ("vulnerability",)),
    ("security disclosure", "gap", "Do you run a bug bounty programme?", ("bug bounty",)),
    # ------------------------------------------------------------------
    # gap — working with StreamFlix
    # ------------------------------------------------------------------
    ("careers", "gap", "How do I apply for an internship at StreamFlix?", ("internship",)),
    ("careers", "gap", "Do you hire subtitle translators?", ("translator", "translators")),
    ("press", "gap", "Who is your press contact?", ("press contact", "journalists")),
    ("press", "gap", "Can journalists get a screener before release?", ("screener", "journalists")),
    ("advertising", "gap", "Can I advertise my product on StreamFlix?", ("advertise",)),
    ("developer API", "gap", "Do you have a public API for developers?", ("api", "developers")),
    ("submissions", "gap", "How does an independent filmmaker submit a short film?", ("filmmaker", "submit")),
    ("distribution licensing", "gap", "How do I license your Originals for an airline?", ("airline",)),
    ("affiliates", "gap", "What commission does the affiliate programme pay?", ("commission", "affiliate")),
    ("affiliates", "gap", "Can influencers get a promo code to share with followers?", ("influencers", "followers")),
    # ------------------------------------------------------------------
    # gap — business, education and public use
    # ------------------------------------------------------------------
    ("hospitality", "gap", "Can a hotel offer StreamFlix in its guest rooms?", ("hotel",)),
    ("commercial premises", "gap", "Do you have a plan for gyms or waiting rooms?", ("gym", "gyms", "waiting room")),
    ("volume licences", "gap", "Can I get a volume licence for 50 employees?", ("employees",)),
    ("campus licences", "gap", "Can my university get a campus licence?", ("university", "campus")),
    ("public screenings", "gap", "Can I screen one of your films at a community event?", ("community event", "screening")),
    ("classroom use", "gap", "Can a teacher show a StreamFlix documentary in class?", ("class", "teacher")),
    # ------------------------------------------------------------------
    # gap — gifts, merchandise and physical media
    # ------------------------------------------------------------------
    ("gift cards", "gap", "Do StreamFlix gift cards expire?", ("gift",)),
    ("gift cards", "gap", "How do I check the balance on a gift card?", ("gift", "balance")),
    ("gift subscriptions", "gap", "Can I buy someone a three-month gift subscription?", ("gift",)),
    ("merchandise", "gap", "Do you sell StreamFlix t-shirts?", ("t-shirts", "merchandise")),
    ("merchandise", "gap", "Can I buy posters of your Originals?", ("posters",)),
    ("soundtracks", "gap", "Where can I buy the soundtrack to one of your Originals?", ("soundtrack",)),
    ("physical media", "gap", "Can I get a Blu-ray box set?", ("blu-ray",)),
    # ------------------------------------------------------------------
    # gap — accessibility
    # ------------------------------------------------------------------
    ("sign language", "gap", "Do any titles have sign language interpretation?", ("sign language",)),
    ("low vision", "gap", "Is there a high-contrast mode for low vision?", ("high-contrast", "low vision")),
    ("braille", "gap", "Does the app work with a braille display?", ("braille",)),
    ("motor access", "gap", "Does the app support switch control for people with motor impairments?", ("switch control", "impairments")),
    ("accessibility statement", "gap", "Where is your WCAG accessibility statement?", ("wcag", "accessibility")),
    # ------------------------------------------------------------------
    # gap — live events and sharing
    # ------------------------------------------------------------------
    ("live sport", "gap", "Can I watch Formula 1 races live?", ("formula",)),
    ("live sport", "gap", "Do you show the Olympics?", ("olympics",)),
    ("live ceremonies", "gap", "Do you stream award ceremonies live?", ("ceremonies", "ceremony")),
    ("sharing enforcement", "gap", "How do you detect account sharing between different homes?", ("account sharing",)),
    ("sharing enforcement", "gap", "Will I be banned if my daughter uses my login at university?", ("banned", "university")),
    ("extra households", "gap", "How much does it cost to add a second home to my account?", ("second home", "add-on")),
    # ------------------------------------------------------------------
    # off_domain — other services (the trap: answering with OUR procedure)
    # ------------------------------------------------------------------
    ("other services", "off_domain", "How do I cancel my Netflix subscription?", ("netflix",)),
    ("other services", "off_domain", "How do I reset my Hulu password?", ("hulu",)),
    ("other services", "off_domain", "How do I get a refund from Amazon Prime Video?", ("prime video",)),
    ("other services 2", "off_domain", "What's new on Disney+ this month?", ("disney",)),
    ("other services 2", "off_domain", "How do I change my Spotify plan?", ("spotify",)),
    # ------------------------------------------------------------------
    # off_domain — things the MODEL knows and the context does not
    # ------------------------------------------------------------------
    ("film trivia", "off_domain", "Who directed Jaws?", ("jaws",)),
    ("film trivia", "off_domain", "What year did Titanic come out?", ("titanic",)),
    ("film trivia", "off_domain", "Is the film Oppenheimer historically accurate?", ("oppenheimer",)),
    ("viewing advice", "off_domain", "Can you recommend a good horror film for tonight?", ("horror",)),
    ("viewing advice", "off_domain", "What should I watch if I liked Inception?", ("inception",)),
    ("geography", "off_domain", "What's the capital of Australia?", ("australia", "capital")),
    ("geography", "off_domain", "What's the population of Canada?", ("population",)),
    ("geography", "off_domain", "What time is it in Tokyo right now?", ("tokyo",)),
    ("sport results", "off_domain", "Who won the World Cup in 2018?", ("world cup",)),
    ("sport results", "off_domain", "What's the cricket score?", ("cricket",)),
    ("politics", "off_domain", "Who is the president of France?", ("president", "france")),
    ("science", "off_domain", "Can you explain quantum entanglement?", ("quantum",)),
    ("science", "off_domain", "How do vaccines work?", ("vaccines", "vaccine")),
    ("arithmetic", "off_domain", "What's the square root of 144?", ("square root",)),
    ("arithmetic", "off_domain", "Convert 5 miles to kilometres for me.", ("kilometres",)),
    ("literature", "off_domain", "What's the plot of Hamlet?", ("hamlet",)),
    ("calendar", "off_domain", "How many days until Christmas?", ("christmas",)),
    # ------------------------------------------------------------------
    # off_domain — tasks a support assistant should not take on
    # ------------------------------------------------------------------
    ("writing tasks", "off_domain", "Write me a poem about autumn.", ("poem", "autumn")),
    ("writing tasks", "off_domain", "Help me write a cover letter for a job.", ("cover letter",)),
    ("writing tasks", "off_domain", "Draft an email to my landlord about the rent.", ("landlord",)),
    ("translation", "off_domain", "Translate 'good morning' into French.", ("french",)),
    ("jokes", "off_domain", "Tell me a joke about cats.", ("joke", "cats")),
    ("cooking", "off_domain", "How do I make sourdough bread?", ("sourdough", "bread")),
    ("cooking", "off_domain", "How many calories are in a banana?", ("calories", "banana")),
    ("home repair", "off_domain", "How do I fix a leaking tap?", ("leaking", "tap")),
    ("cars", "off_domain", "How do I change a car tyre?", ("tyre",)),
    ("health", "off_domain", "How should I treat sunburn?", ("sunburn",)),
    ("fitness", "off_domain", "Give me a seven-day workout plan.", ("workout",)),
    ("pets", "off_domain", "How do I teach my dog to sit?", ("dog",)),
    ("money", "off_domain", "What's a good mortgage rate at the moment?", ("mortgage",)),
    ("money", "off_domain", "How do I invest in index funds?", ("index funds", "invest")),
    ("money", "off_domain", "How do I file my income tax return?", ("income tax", "tax return")),
    ("travel", "off_domain", "Book me a flight to Madrid.", ("flight", "madrid")),
    ("travel", "off_domain", "What's the exchange rate for euros today?", ("exchange rate", "euros")),
    ("shopping advice", "off_domain", "Should I buy an OLED or a QLED television?", ("oled", "qled")),
    ("shopping advice", "off_domain", "Which broadband provider is fastest in my area?", ("broadband",)),
    ("tech support elsewhere", "off_domain", "How do I change the admin password on my router?", ("admin",)),
    ("tech support elsewhere", "off_domain", "How do I convert an MKV file to MP4?", ("mkv", "mp4")),
    ("tech support elsewhere", "off_domain", "How do I jailbreak my Fire Stick?", ("jailbreak",)),
    ("books", "off_domain", "Can you recommend a book on machine learning?", ("machine learning", "book")),
    ("miscellany", "off_domain", "What's my horoscope for today?", ("horoscope",)),
]

TIERS = ("near_miss", "gap", "off_domain")
MAX_PER_TOPIC = 3


def load_oos_extension() -> list[dict]:
    """The extension set, in the same shape the golden set uses.

    `gt_article_ids` is empty for every question: nothing in the corpus
    answers them, which is the whole point.
    """
    return [
        {
            "question_id": f"oosx-{i:03d}",
            "question": question,
            "category": "out_of_scope",
            "gt_article_ids": [],
            "reference_answer": "",
            "topic": topic,
            "tier": tier,
            "absent_terms": list(terms),
            "notes": f"EXTENSION ({tier}): {topic}.",
        }
        for i, (topic, tier, question, terms) in enumerate(_QUESTIONS, 1)
    ]
