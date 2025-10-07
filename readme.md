Role

-   You are an expert front-end ML engineer building a browser-based Two-Tower retrieval demo with TensorFlow.js for the MovieLens 100K dataset (u.data, u.item), featuring Deep Learning comparison, genre features, and user features, suitable for static GitHub Pages hosting.


Context

-   Dataset: MovieLens 100K
    
    -   u.data format: user_id, item_id, rating, timestamp separated by tabs; 100k interactions; 943 users; 1,682 items.
        
    -   u.item format: item_id|title|release_date|...|genre_features; use item_id, title, year parsed from title, and 19 genre features (last 19 fields are binary genre indicators).
        
-   Goal: Build an in-browser Two-Tower model with Deep Learning comparison:
    
    -   Simple Two-Tower: user_id → embedding, item_id → embedding, dot product scoring
        
    -   Deep Two-Tower: user_id + user_features → MLP → embedding, item_id + genre_features → MLP → embedding, dot product scoring
        
    -   Loss: sampled-softmax (in-batch negatives) with Adam optimizer
        
    -   Comparison: Train both models simultaneously and compare recommendations side-by-side
        
-   Enhanced Features:
    
    -   Genre Features: Extract 19 binary genre indicators from u.item (Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western)
        
    -   User Features: Generate user preference features based on genre preferences, average rating, and rating count
        
    -   Deep Learning: MLP with at least one hidden layer in both User and Item towers
        
    -   Feature Processing: Separate dense layers for processing genre and user features before combining with base embeddings
        
-   UX requirements:
    
    -   Buttons: "Load Data", "Train", "Test" (in Russian: "Загрузить данные", "Обучить модели", "Тестировать").
        
    -   Training shows live loss chart for Deep Learning model and epoch progress for both models.
        
    -   After training, render 2D projection (PCA via numeric approximation) of item embeddings from Deep Learning model.
        
    -   Test action: randomly select a user who has at least 20 ratings; show three tables:
        
        -   Left: user's top-10 historically rated movies (by rating, then recency) with genre information
            
        -   Middle: simple model's top-10 recommended movies with scores and genres
            
        -   Right: deep learning model's top-10 recommended movies with scores and genres
            
    -   Present the three lists in a CSS Grid layout with modern styling.
        
    -   Display genre names for each movie in all tables.
        
-   Constraints:
    
    -   Pure client-side (no server), runs on GitHub Pages. Fetch u.data and u.item via relative paths (place files under data/).
        
    -   Use TensorFlow.js only; no Python, no build step.
        
    -   Keep memory in check: allow limiting interactions (e.g., max 80k) and embedding dim (e.g., 32).
        
    -   Deterministic seeding optional; browsers vary.
        
    -   Support Russian interface with English technical terms.
        
-   References for correctness:
    
    -   Two-tower retrieval on MovieLens in TF/TFRS (concepts and loss)
        
    -   MovieLens 100K format details and genre structure
        
    -   TensorFlow.js in-browser training guidance


Instructions

-   Return three files with complete code, each in a separate fenced code block.
    
-   Implement clean, commented JavaScript with clear sections and Russian UI elements.
    
-   Include comprehensive genre and user feature processing.


a) index.html

-   Include:
    
    -   Title "Two-Tower Movie Recommender с Deep Learning" and modern CSS with CSS Grid.
        
    -   Buttons: "Загрузить данные", "Обучить модели", "Тестировать".
        
    -   Status area with blue accent styling, loss chart canvas, and embedding projection canvas.
        
    -   A <div id="results"> to hold the three-column comparison table (historical ratings, simple model, deep model).
        
    -   Scripts: load TensorFlow.js from CDN, then app.js and two-tower.js.
        
    -   CSS Grid layout for three-column comparison with responsive design.
        
    -   Modern styling with hover effects, rounded corners, and professional color scheme.
        
    -   Genre information display in table columns.
        
-   Add usability tips (how long training takes, how to host files on GitHub Pages).


b) app.js

-   Data loading:
    
    -   Fetch data/u.data and data/u.item with fetch(); parse lines; build:
        
        -   interactions: [{userId, itemId, rating, timestamp}]
            
        -   items: Map itemId → {title, year, genres: [19 binary values]}
            
    -   Extract genre features from last 19 fields of u.item (binary genre indicators).
        
    -   Create user features based on genre preferences, average rating, and rating count.
        
    -   Build user→rated items and user→top-rated (compute once).
        
    -   Create integer indexers for userId and itemId to 0-based indices; store reverse maps.
        
    -   Create genre feature matrix and user feature matrix for model input.
        
-   Train pipeline:
    
    -   Initialize both SimpleTwoTowerModel and DeepTwoTowerModel simultaneously.
        
    -   Train both models in parallel: for each (u, i_pos), use in-batch negatives.
        
    -   Show live loss chart for Deep Learning model with both simple and deep loss values in status.
        
    -   Allow config: epochs, batch size, embeddingDim, learningRate, maxInteractions.
        
    -   Display training progress for both models in status messages.
        
-   Test pipeline:
    
    -   Pick a random user with ≥20 ratings.
        
    -   Compute user embeddings from both models; compute scores vs all items using matrix ops.
        
    -   Exclude items the user already rated; return top-10 titles from both models.
        
    -   Render three-column HTML table: historical ratings, simple model recommendations, deep model recommendations.
        
    -   Include genre information for each movie in all tables.
        
-   Visualization:
    
    -   After training, take a sample (e.g., 500 items), project item embeddings from Deep Learning model to 2D with PCA.
        
    -   Draw scatter plot with modern styling and proper labels.
        
-   Feature processing:
    
    -   createGenreFeatures(): Extract and normalize genre features from u.item.
        
    -   createUserFeatures(): Generate user preference features based on high-rated movies (rating >= 4).
        
    -   getGenreNames(): Helper function to convert binary genre vectors to readable genre names.


c) two-tower.js

-   Implement two Two-Tower models in TF.js:
    
    -   Class SimpleTwoTowerModel (baseline without deep learning):
        
        -   constructor(numUsers, numItems, embeddingDim)
            
        -   userForward(userIdxTensor) → embeddings gather
            
        -   itemForward(itemIdxTensor) → embeddings gather
            
        -   score(uEmb, iEmb): dot product along last dim
            
    -   Class DeepTwoTowerModel (with MLP and features):
        
        -   constructor(numUsers, numItems, embeddingDim, genreFeatures, userFeatures)
            
        -   User Tower: userEmbeddingTable + userFeatures → MLP → embedding
            
        -   Item Tower: itemEmbeddingTable + genreFeatures → MLP → embedding
            
        -   Feature processing layers for both genre and user features
            
        -   MLP with hidden layer (embeddingDim * 2 units) and output layer (embeddingDim units)
            
        -   L2 normalization for stable embeddings
            
    -   Loss (both models):
        
        -   In-batch sampled softmax: For a batch of user embeddings U and positive item embeddings I+, compute logits = U @ I^T, labels = diagonal; apply softmax cross-entropy.
            
        -   Adam optimizer with gradient tape to update all parameters.
            
        -   Return scalar loss for UI plotting.
            
    -   Training step:
        
        -   trainStep(userIndices, itemIndices): Process batch through forward pass, compute loss, apply gradients.
            
        -   Return scalar loss for UI plotting.
            
    -   Inference:
        
        -   getUserEmbedding(userIndex): Get user embedding for inference.
            
        -   getScoresForAllItems(userEmbedding): Compute scores vs all items with batched operations.
            
        -   getItemEmbeddings(): Get all item embeddings for visualization.
            
-   Comments:
    
    -   Add comprehensive comments explaining two-tower architecture, MLP design, feature processing, and in-batch negatives.
        
    -   Explain genre feature integration and user feature generation.
        
    -   Document the comparison between simple and deep models.


Format

-   Return three code blocks only, labeled exactly:
    
    -   index.html
        
    -   app.js
        
    -   two-tower.js
        
-   No extra prose outside the code blocks.
    
-   Ensure the code runs when the repository structure is:
    
    -   /index.html
        
    -   /app.js
        
    -   /two-tower.js
        
    -   /data/u.data
        
    -   /data/u.item
        
-   The UI must:
    
    -   Load Data → parse and index data, extract genre and user features.
        
    -   Train → run epochs for both models, update loss chart, then draw embedding projection.
        
    -   Test → pick a random qualified user, render three-column comparison table with genre information.
        
-   Additional Requirements:
    
    -   Russian interface with professional English technical terminology.
        
    -   Modern CSS Grid layout for three-column comparison.
        
    -   Genre information display in all recommendation tables.
        
    -   Comprehensive feature processing for both genre and user attributes.
        
    -   Deep Learning model with MLP architecture and feature integration.
        
    -   Side-by-side comparison of simple vs deep learning recommendations.
