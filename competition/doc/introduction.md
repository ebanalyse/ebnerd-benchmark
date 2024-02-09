# Introduction to EBNeRD

## Overall Introduction

The EBNeRD dataset for news recommendation was collected from anonymized behavior logs of <a href="https://eb.dk/">Ekstra Bladet</a> website. (...)


## Dataset Format

Both the training, validation, and test data are a zip-compressed folder, which contains X different files:

File Name | Description
-------------   | -------------
behaviors.parquet          | The impression logs of users
history.parquet            | The click histories of users
articles.parquet            | The information of news articles
document_embeddings.parquet | The embeddings of the articles textual information
image_embeddings.parquet    | The embeddings of the articles visual information **(TBA)**

### behaviors.parquet

The *behaviors.parquet* file contains the impression logs. 
It has 17 columns:
- Impression ID: The ID of an impression.
- User ID: The anonymous ID of a user.
- Article ID: The unique ID of a news article. An empty field means the impression is from the front page.
- Session ID: A unique identifier for a user's browsing session.
- Inview Article IDs. List of news displayed in this impression. The orders of news in a impressions have been shuffled. 
- Clicked Article IDs:  List of news clicked in this impression.
- Time: The impression time with format "YYYY/MM/DD HH:MM:SS".
- Readtime: The amount of seconds a user spends on a given page.
- Scroll Percentage: The percentage of an article that a user scrolls through, indicating how much of the content was potentially viewed.
- Device Type: The type of device used to access the content, such as desktop (1) mobile (2), tablet (3), or unknown (0).
- SSO Status: Indicates whether a user is logged in through Single Sign-On (SSO) authentication.
- Subscription Status: The user's subscription status, indicating whether they are a paid subscriber.
- Gender: The gender of the user, either Male (0) or Female (1), as specified in their profile.
- Postcode: The user's postcode, aggregated at the district level as specified in their profile, with metropolitan (0), rural district (1), municipality (2), provincial (3), big city (4).
- Age: The age of the user, as specified in their profile, categorized into bins of 10 years (e.g., 20-29, 30-39 etc.).
- Next Readtime: The time a user spends on the next clicked article, i.e., the article in clicked Article IDs.
- Next Scroll Percentage: The scroll percentage for a user's next article interaction, i.e., the article in clicked Article IDs.

An example is shown in the table below:

Column | dtype | Content
------------- | ------------- | -------------
Impression ID           | u32           |   153
User ID                 | u32           |   44038 
Article ID              | i32           |   9650148
Session ID              | u32           |   1153
Inview Article IDs      | list[i32]     |   [9649538, 9649689, … 9649569]
Clicked Article IDs     | list[i32]     |   [9649689]
Time                    | datetime[μs]  |   14.0
Readtime                | f32           |   2023-02-25 06:41:40
Scroll Percentage       | d32           |   100.0
Device Type             | i8            |   1
SSO Status              | bool          |   True
Subscription Status     | bool          |   True
Gender                  | i8            |   null
Postcode                | i8            |   2
Age                     | i8            |   50
Next Readtime           | f32           |   8.0
Next Scroll Percentage  | f32           |   41.0


### history.parquet

The *history.parquet* file contains the click histories of users.
It has 5 columns:
- User ID: The anonymous ID of a user.
- Article IDs: The articles clicked by the user.
- Timestamps: The timestamps of when the articles were clicked.
- Read Times: The read times of the clicked articles.
- Scroll Percentages: The scroll percentages of the clicked articles.

An example is shown in the table below:

Column | dtype | Content
------------- | ------------- | -------------
User ID             | u32                   |   44038
Article IDs         | list[i32]             |   [9618533, … 9646154]
Timestamps          | list[datetime[μs]]    |   [2023-02-02 16:37:42, … 2023-02-22 18:28:38] 
Read times          | list[f32]             |   [425.0, … 12.0]
Scroll Percentages  | list[f32]             |   [null, … 100.0]

### articles.parquet

The *articles.parquet* file contains the detailed information of news articles involved in the behaviors.parquet file.
It has 16 columns:
- Article ID: The article's ID.
- Title: The article's Danish title.
- Abstract: The article's Danish subtitle.
- Body: The article's full Danish text body.
- Category ID: The category ID.
- Category String: The category in written form.
- SubCategory IDs: The subcategories.
- Premium: Whether the content is behind a paywall.
- Time Published: The time the article was published.
- Time Modified: The timestamp for the last modification of the article, e.g., updates as the story evolves or spelling corrections.
- Image IDs: The image IDs used in the article.
- Article Type: The type of article, such as a feature, gallery, video, or live blog.
- URL: The article's URL.
- NER: The tags retrieved from a proprietary named-entity-recognition model at Ekstra Bladet, based on the concatenated title, abstract, and body.
- Entities: The tags retrieved from a proprietary entity-recognition model at Ekstra Bladet, based on the concatenated title, abstract, and body.
- Topics: The tags retrieved from a proprietary topic-recognition model at Ekstra Bladet, based on the concatenated title, abstract, and body.

An example is shown in the following table:

Column  | dtype | Content
------------- | ------------- | -------------
Article ID        | i32             |   8987932
Title             | str             |   Se billederne: Zlatans paradis til salg
Abstract          | str             |   Zlatan Ibrahimovic har sat sin skihytte i Åre til salg, men prisen skal nok afskrække en del. (...)
Body              | str             |   Drømmer du om en eksklusiv skihytte i Sverige? Så har Zlatan Ibrahimovic et eksklusivt tilbud til dig (...)
Category ID       | i16             |   142
Category String   | list[i16]       |   sport
SubCategory IDs   | str             |   [196, 271]
Premium:          | bool            |   False
Time Published    | datetime[μs]    |   2021-11-15 03:56:56
Time Modified     | datetime[μs]    |   2023-06-29 06:38:41
Image IDs         | list[i64]       |   [8988118]
Article Type      | str             |   article_default
URL               | str             |   https://ekstrabladet.dk/sport/fodbold/landsholdsfodbold/se-billederne-zlatans-paradis-til-salg/8987932 
NER               | list[str]	    |   ['Aftonbladet', 'Åre', 'Bjurfors', 'Cecilia Edfeldt Jigstedt', 'Helena', 'Sverige', 'Zlatan Ibrahimovic']
Entities          | list[str]	    |   ['ORG', 'LOC', 'ORG', 'PER', 'PER', 'LOC', 'PER']
Topics            | list[str]	    |   []


### document_embeddings.parquet & image_embeddings.parquet
To initiate the quick use of EBNeRD, the dataset features embedding artifacts, i.e., the textual representation and the encoded thumbnail images. We also provide an example of how to easily generate your own document embeddings using [Hugging Face](https://huggingface.co/).

The artifacts follows the example:

ID | dtype | Embedding Values
------------- | ------------- | -------------
i32 | list[f64] | [0.062796, 0.040298, … 0.006439]

