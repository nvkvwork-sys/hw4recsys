class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();
        this.userTopRated = new Map();
        this.simpleModel = null;  // Простая модель без глубокого обучения
        this.deepModel = null;    // Глубокая модель с MLP
        this.genreFeatures = null; // Признаки жанров для фильмов
        this.userFeatures = null;  // Пользовательские признаки
        this.genreNames = [
            'Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ];
        
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 20,
            learningRate: 0.001
        };
        
        this.lossHistory = [];
        this.isTraining = false;
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
        this.updateStatus('Click "Load Data" to start');
    }
    
    async loadData() {
        this.updateStatus('Loading data...');
        
        try {
            // Load interactions
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            const interactionsLines = interactionsText.trim().split('\n');
            
            this.interactions = interactionsLines.slice(0, this.config.maxInteractions).map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });
            
            // Load items
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            const itemsLines = itemsText.trim().split('\n');
            
            itemsLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                // Извлекаем признаки жанров (последние 19 полей)
                const genreFeatures = parts.slice(-19).map(val => parseInt(val));
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: genreFeatures
                });
            });
            
            // Create mappings and find users with sufficient ratings
            this.createMappings();
            
            this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.qualifiedUsers.length} users have 20+ ratings.`);
            
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }
    
    createMappings() {
        // Create user and item mappings to 0-based indices
        const userSet = new Set(this.interactions.map(i => i.userId));
        const itemSet = new Set(this.interactions.map(i => i.itemId));
        
        Array.from(userSet).forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        
        Array.from(itemSet).forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
        
        // Group interactions by user and find top rated movies
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!userInteractions.has(userId)) {
                userInteractions.set(userId, []);
            }
            userInteractions.get(userId).push(interaction);
        });
        
        // Sort each user's interactions by rating (desc) and timestamp (desc)
        userInteractions.forEach((interactions, userId) => {
            interactions.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });
        
        this.userTopRated = userInteractions;
        
        // Создаем матрицу признаков жанров для всех фильмов
        this.createGenreFeatures();
        
        // Создаем пользовательские признаки на основе их предпочтений по жанрам
        this.createUserFeatures();
        
        // Находим пользователей с достаточным количеством оценок
        this.findQualifiedUsers();
    }
    
    createGenreFeatures() {
        // Создаем матрицу признаков жанров для всех фильмов
        const itemIds = Array.from(this.itemMap.keys()).sort((a, b) => a - b);
        this.genreFeatures = [];
        
        itemIds.forEach(itemId => {
            const item = this.items.get(itemId);
            if (item && item.genres) {
                this.genreFeatures.push(item.genres);
            } else {
                // Если жанры не найдены, используем нулевой вектор
                this.genreFeatures.push(new Array(19).fill(0));
            }
        });
        
        console.log(`Created ${this.genreFeatures.length} genre feature vectors`);
    }
    
    createUserFeatures() {
        // Создаем пользовательские признаки на основе их предпочтений по жанрам
        const userIds = Array.from(this.userMap.keys()).sort((a, b) => a - b);
        this.userFeatures = [];
        
        userIds.forEach(userId => {
            const userInteractions = this.userTopRated.get(userId) || [];
            
            // Инициализируем вектор предпочтений по жанрам
            const genrePreferences = new Array(19).fill(0);
            let totalRatings = 0;
            
            // Анализируем жанры фильмов, которые пользователь высоко оценил (рейтинг >= 4)
            userInteractions.forEach(interaction => {
                if (interaction.rating >= 4) {
                    const item = this.items.get(interaction.itemId);
                    if (item && item.genres) {
                        item.genres.forEach((hasGenre, genreIndex) => {
                            if (hasGenre === 1) {
                                genrePreferences[genreIndex] += interaction.rating;
                                totalRatings += interaction.rating;
                            }
                        });
                    }
                }
            });
            
            // Нормализуем предпочтения
            if (totalRatings > 0) {
                for (let i = 0; i < genrePreferences.length; i++) {
                    genrePreferences[i] = genrePreferences[i] / totalRatings;
                }
            }
            
            // Добавляем дополнительные признаки
            const avgRating = userInteractions.length > 0 ? 
                userInteractions.reduce((sum, i) => sum + i.rating, 0) / userInteractions.length : 0;
            const numRatings = userInteractions.length;
            
            // Объединяем все признаки
            const userFeatureVector = [
                ...genrePreferences,
                avgRating / 5.0, // Нормализованная средняя оценка
                Math.min(numRatings / 100.0, 1.0) // Нормализованное количество оценок
            ];
            
            this.userFeatures.push(userFeatureVector);
        });
        
        console.log(`Created ${this.userFeatures.length} user feature vectors`);
    }
    
    findQualifiedUsers() {
        // Filter users with at least 20 ratings
        const qualifiedUsers = [];
        this.userTopRated.forEach((interactions, userId) => {
            if (interactions.length >= 20) {
                qualifiedUsers.push(userId);
            }
        });
        this.qualifiedUsers = qualifiedUsers;
    }
    
    async train() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('train').disabled = true;
        this.lossHistory = [];
        
        try {
            this.updateStatus('Initializing models...');
            
            // Clear existing models to avoid variable name conflicts
            if (this.simpleModel) {
                this.simpleModel.dispose();
                this.simpleModel = null;
            }
            if (this.deepModel) {
                this.deepModel.dispose();
                this.deepModel = null;
            }
            
            // Clear TensorFlow.js memory
            tf.dispose();
        
        // Validate data before training
        if (!this.interactions || this.interactions.length === 0) {
            throw new Error('No interactions loaded');
        }
        if (!this.genreFeatures || this.genreFeatures.length === 0) {
            throw new Error('No genre features created');
        }
        if (!this.userFeatures || this.userFeatures.length === 0) {
            throw new Error('No user features created');
        }
        
        // Initialize simple model (without deep learning)
        try {
            this.simpleModel = new SimpleTwoTowerModel(
                this.userMap.size,
                this.itemMap.size,
                this.config.embeddingDim
            );
        } catch (error) {
            console.error('Simple model initialization failed:', error.message);
            throw error;
        }
        
        // Initialize deep model (with MLP, genre features, and user features)
        console.log('Initializing deep model with:', {
            numUsers: this.userMap.size,
            numItems: this.itemMap.size,
            embeddingDim: this.config.embeddingDim,
            genreFeaturesLength: this.genreFeatures ? this.genreFeatures.length : 0,
            userFeaturesLength: this.userFeatures ? this.userFeatures.length : 0
        });
        
        try {
            this.deepModel = new DeepTwoTowerModel(
                this.userMap.size,
                this.itemMap.size,
                this.config.embeddingDim,
                this.genreFeatures,
                this.userFeatures
            );
        } catch (error) {
            console.warn('Deep model failed, using simple model instead:', error.message);
            // Fallback to simple model if deep model fails
            this.deepModel = new SimpleTwoTowerModel(
                this.userMap.size,
                this.itemMap.size,
                this.config.embeddingDim
            );
        }
        
        console.log('Models initialized successfully');
        
        // Prepare training data
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        
        this.updateStatus('Starting training...');
        
        // Training loop for both models
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let simpleEpochLoss = 0;
            let deepEpochLoss = 0;
            
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                
                const batchUsers = userIndices.slice(start, end);
                const batchItems = itemIndices.slice(start, end);
                
                let simpleLoss = 0;
                let deepLoss = 0;
                
                try {
                    // Train simple model first
                    console.log(`Training simple model - Epoch ${epoch + 1}, Batch ${batch + 1}`);
                    simpleLoss = await this.simpleModel.trainStep(batchUsers, batchItems);
                    simpleEpochLoss += simpleLoss;
                    
                    // Train deep model
                    console.log(`Training deep model - Epoch ${epoch + 1}, Batch ${batch + 1}`);
                    deepLoss = await this.deepModel.trainStep(batchUsers, batchItems);
                    deepEpochLoss += deepLoss;
                    
                    // Use deep model loss for chart (more interesting to track)
                    this.lossHistory.push(deepLoss);
                    this.updateLossChart();
                    
                } catch (error) {
                    console.error('Training error:', error);
                    this.updateStatus(`Training error: ${error.message}`);
                    throw error; // Re-throw to be caught by outer try-catch
                }
                
                if (batch % 10 === 0) {
                    this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}, Simple Loss: ${simpleLoss.toFixed(4)}, Deep Loss: ${deepLoss.toFixed(4)}`);
                }
                
                // Allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            
            simpleEpochLoss /= numBatches;
            deepEpochLoss /= numBatches;
            this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs} completed. Simple Avg Loss: ${simpleEpochLoss.toFixed(4)}, Deep Avg Loss: ${deepEpochLoss.toFixed(4)}`);
        }
        
            this.isTraining = false;
            document.getElementById('train').disabled = false;
            document.getElementById('test').disabled = false;
            
            this.updateStatus('Training completed! Click "Test" to see recommendations.');
            
            // Visualize embeddings
            this.visualizeEmbeddings();
        } catch (error) {
            console.error('Training failed:', error);
            this.updateStatus(`Training failed: ${error.message}`);
            this.isTraining = false;
            document.getElementById('train').disabled = false;
        }
    }
    
    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.lossHistory.length === 0) return;
        
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;
        
        ctx.strokeStyle = '#007acc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.lossHistory.forEach((loss, index) => {
            const x = (index / this.lossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, canvas.height - 10);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 20);
    }
    
    async visualizeEmbeddings() {
        if (!this.deepModel) return;
        
        this.updateStatus('Computing embedding visualization...');
        
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            // Sample items for visualization
            const sampleSize = Math.min(500, this.itemMap.size);
            const sampleIndices = Array.from({length: sampleSize}, (_, i) => 
                Math.floor(i * this.itemMap.size / sampleSize)
            );
            
            // Get embeddings and compute PCA
            const embeddingsTensor = this.deepModel.getItemEmbeddings();
            const embeddings = embeddingsTensor.arraySync();
            const sampleEmbeddings = sampleIndices.map(i => embeddings[i]);
            
            const projected = this.computePCA(sampleEmbeddings, 2);
            
            // Normalize to canvas coordinates
            const xs = projected.map(p => p[0]);
            const ys = projected.map(p => p[1]);
            
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            const yMin = Math.min(...ys);
            const yMax = Math.max(...ys);
            
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            
            // Draw points
            ctx.fillStyle = 'rgba(0, 122, 204, 0.6)';
            sampleIndices.forEach((itemIdx, i) => {
                const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 40) + 20;
                const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 40) + 20;
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
            
            // Add title and labels
            ctx.fillStyle = '#000';
            ctx.font = '14px Arial';
            ctx.fillText('Item Embeddings Projection (PCA)', 10, 20);
            ctx.font = '12px Arial';
            ctx.fillText(`Showing ${sampleSize} items`, 10, 40);
            
            this.updateStatus('Embedding visualization completed.');
        } catch (error) {
            this.updateStatus(`Error in visualization: ${error.message}`);
        }
    }
    
    computePCA(embeddings, dimensions) {
        // Simple PCA using power iteration
        const n = embeddings.length;
        const dim = embeddings[0].length;
        
        // Center the data
        const mean = Array(dim).fill(0);
        embeddings.forEach(emb => {
            emb.forEach((val, i) => mean[i] += val);
        });
        mean.forEach((val, i) => mean[i] = val / n);
        
        const centered = embeddings.map(emb => 
            emb.map((val, i) => val - mean[i])
        );
        
        // Compute covariance matrix
        const covariance = Array(dim).fill(0).map(() => Array(dim).fill(0));
        centered.forEach(emb => {
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] += emb[i] * emb[j];
                }
            }
        });
        covariance.forEach(row => row.forEach((val, j) => row[j] = val / n));
        
        // Power iteration for first two components
        const components = [];
        for (let d = 0; d < dimensions; d++) {
            let vector = Array(dim).fill(1/Math.sqrt(dim));
            
            for (let iter = 0; iter < 10; iter++) {
                let newVector = Array(dim).fill(0);
                
                for (let i = 0; i < dim; i++) {
                    for (let j = 0; j < dim; j++) {
                        newVector[i] += covariance[i][j] * vector[j];
                    }
                }
                
                const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
                vector = newVector.map(val => val / norm);
            }
            
            components.push(vector);
            
            // Deflate the covariance matrix
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] -= vector[i] * vector[j];
                }
            }
        }
        
        // Project data
        return embeddings.map(emb => {
            return components.map(comp => 
                emb.reduce((sum, val, i) => sum + val * comp[i], 0)
            );
        });
    }
    
    async test() {
        if (!this.simpleModel || !this.deepModel || this.qualifiedUsers.length === 0) {
            this.updateStatus('Models not trained or no qualified users found.');
            return;
        }
        
        this.updateStatus('Generating recommendations...');
        
        try {
            // Pick random qualified user
            const randomUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
            const userInteractions = this.userTopRated.get(randomUser);
            const userIndex = this.userMap.get(randomUser);
            
            // Get user embeddings from both models
            const simpleUserEmb = this.simpleModel.getUserEmbedding(userIndex);
            const deepUserEmb = this.deepModel.getUserEmbedding(userIndex);
            
            // Get scores for all items from both models
            const simpleItemScores = await this.simpleModel.getScoresForAllItems(simpleUserEmb);
            const deepItemScores = await this.deepModel.getScoresForAllItems(deepUserEmb);
            
            // Filter out items the user has already rated
            const ratedItemIds = new Set(userInteractions.map(i => i.itemId));
            const simpleCandidateScores = [];
            const deepCandidateScores = [];
            
            simpleItemScores.forEach((score, itemIndex) => {
                const itemId = this.reverseItemMap.get(itemIndex);
                if (!ratedItemIds.has(itemId)) {
                    simpleCandidateScores.push({ itemId, score, itemIndex });
                }
            });
            
            deepItemScores.forEach((score, itemIndex) => {
                const itemId = this.reverseItemMap.get(itemIndex);
                if (!ratedItemIds.has(itemId)) {
                    deepCandidateScores.push({ itemId, score, itemIndex });
                }
            });
            
            // Sort by score descending and take top 10
            simpleCandidateScores.sort((a, b) => b.score - a.score);
            deepCandidateScores.sort((a, b) => b.score - a.score);
            
            const simpleTopRecommendations = simpleCandidateScores.slice(0, 10);
            const deepTopRecommendations = deepCandidateScores.slice(0, 10);
            
            // Display results with comparison
            this.displayComparisonResults(randomUser, userInteractions, simpleTopRecommendations, deepTopRecommendations);
            
        } catch (error) {
            this.updateStatus(`Error generating recommendations: ${error.message}`);
        }
    }
    
    displayComparisonResults(userId, userInteractions, simpleRecommendations, deepRecommendations) {
        const resultsDiv = document.getElementById('results');
        
        const topRated = userInteractions.slice(0, 10);
        
        // Функция для получения названий жанров
        const getGenreNames = (itemId) => {
            const item = this.items.get(itemId);
            if (!item || !item.genres) return 'N/A';
            
            const genres = [];
            item.genres.forEach((hasGenre, index) => {
                if (hasGenre === 1 && this.genreNames[index]) {
                    genres.push(this.genreNames[index]);
                }
            });
            return genres.length > 0 ? genres.join(', ') : 'Unknown';
        };
        
        let html = `
            <h2>Recommendation Comparison for User ${userId}</h2>
            <div class="comparison-container">
                <div class="column">
                    <h3>Top 10 Rated Movies (Historical)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie</th>
                                <th>Rating</th>
                                <th>Year</th>
                                <th>Genres</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        topRated.forEach((interaction, index) => {
            const item = this.items.get(interaction.itemId);
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.title}</td>
                    <td>${interaction.rating}</td>
                    <td>${item.year || 'N/A'}</td>
                    <td>${getGenreNames(interaction.itemId)}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
                <div class="column">
                    <h3>Top 10 Recommendations (Without Deep Learning)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie</th>
                                <th>Score</th>
                                <th>Year</th>
                                <th>Genres</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        simpleRecommendations.forEach((rec, index) => {
            const item = this.items.get(rec.itemId);
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.title}</td>
                    <td>${rec.score.toFixed(4)}</td>
                    <td>${item.year || 'N/A'}</td>
                    <td>${getGenreNames(rec.itemId)}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
                <div class="column">
                    <h3>Top 10 Recommendations (Deep Learning + Genres + User Features)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie</th>
                                <th>Score</th>
                                <th>Year</th>
                                <th>Genres</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        deepRecommendations.forEach((rec, index) => {
            const item = this.items.get(rec.itemId);
            html += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${item.title}</td>
                    <td>${rec.score.toFixed(4)}</td>
                    <td>${item.year || 'N/A'}</td>
                    <td>${getGenreNames(rec.itemId)}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        this.updateStatus('Recommendations generated successfully! Comparison shows the difference between simple model and deep learning model with genres and user features.');
    }
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
