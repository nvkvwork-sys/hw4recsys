// Простая модель Two Towers без глубокого обучения (базовая)
class SimpleTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        
        // Initialize embedding tables with small random values
        // Two-tower architecture: separate user and item embeddings
        const timestamp = Date.now();
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            `simple_user_embeddings_${timestamp}`
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            `simple_item_embeddings_${timestamp}`
        );
        
        // Adam optimizer for stable training
        this.optimizer = tf.train.adam(0.001);
    }
    
    // User tower: simple embedding lookup
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    // Item tower: simple embedding lookup  
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    // Scoring function: dot product between user and item embeddings
    // Dot product is efficient and commonly used in retrieval systems
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            
            // In-batch sampled softmax loss:
            // Use all items in batch as negatives for each user
            // Diagonal elements are positive pairs
            const loss = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor);
                
                // Compute similarity matrix: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positives
                // Use int32 tensor for oneHot indices
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss
                // This encourages positive pairs to have higher scores than negatives
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            // Compute gradients and update embeddings
            const { value, grads } = this.optimizer.computeGradients(loss);
            
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Compute dot product with all item embeddings
            const scores = tf.dot(this.itemEmbeddings, userEmbedding);
            return scores.dataSync();
        });
    }
    
    getItemEmbeddings() {
        // Return the tensor directly - call arraySync() on the tensor, not this method
        return this.itemEmbeddings;
    }
    
    dispose() {
        // Clean up TensorFlow.js resources
        if (this.userEmbeddings) {
            this.userEmbeddings.dispose();
        }
        if (this.itemEmbeddings) {
            this.itemEmbeddings.dispose();
        }
    }
}

// Упрощенная глубокая модель Two Towers с базовыми признаками
class DeepTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, genreFeatures = null, userFeatures = null) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.genreFeatures = genreFeatures; // Матрица признаков жанров для каждого фильма
        this.userFeatures = userFeatures;   // Матрица пользовательских признаков
        this.numGenres = genreFeatures ? genreFeatures[0].length : 0;
        this.numUserFeatures = userFeatures ? userFeatures[0].length : 0;
        
        // User Tower: простая версия с дополнительными признаками
        const timestamp = Date.now();
        this.userEmbeddingTable = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            `deep_user_embedding_table_${timestamp}`
        );
        
        // Простой слой для обработки пользовательских признаков
        if (this.numUserFeatures > 0) {
            this.userFeatureWeight = tf.variable(
                tf.randomNormal([this.numUserFeatures, embeddingDim], 0, 0.01),
                true,
                `user_feature_weight_${timestamp}`
            );
        }
        
        // Item Tower: простая версия с дополнительными признаками
        this.itemEmbeddingTable = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            `deep_item_embedding_table_${timestamp}`
        );
        
        // Простой слой для обработки признаков жанров
        if (this.numGenres > 0) {
            this.genreWeight = tf.variable(
                tf.randomNormal([this.numGenres, embeddingDim], 0, 0.01),
                true,
                `genre_weight_${timestamp}`
            );
        }
        
        // Adam optimizer для глубокого обучения
        this.optimizer = tf.train.adam(0.001);
        
        // Создаем тензоры для жанров и пользовательских признаков один раз для эффективности
        if (this.genreFeatures) {
            this.genreFeaturesTensor = tf.tensor2d(this.genreFeatures);
        }
        if (this.userFeatures) {
            this.userFeaturesTensor = tf.tensor2d(this.userFeatures);
        }
    }
    
    // User Tower: простая версия с дополнительными признаками
    userForward(userIndices) {
        return tf.tidy(() => {
            // Получаем базовые эмбеддинги пользователей
            const baseEmbeddings = tf.gather(this.userEmbeddingTable, userIndices);
            
            let combinedFeatures = baseEmbeddings;
            
            // Добавляем пользовательские признаки если они доступны
            if (this.userFeatures && this.numUserFeatures > 0 && this.userFeatureWeight) {
                const userFeatureEmbeddings = tf.gather(this.userFeaturesTensor, userIndices);
                const processedUserFeatures = tf.matMul(userFeatureEmbeddings, this.userFeatureWeight);
                
                // Объединяем базовые эмбеддинги с обработанными пользовательскими признаками
                combinedFeatures = tf.add(baseEmbeddings, processedUserFeatures);
            }
            
            // Простая нормализация
            return tf.div(combinedFeatures, tf.norm(combinedFeatures, 'euclidean', -1, true));
        });
    }
    
    // Item Tower: простая версия с дополнительными признаками
    itemForward(itemIndices) {
        return tf.tidy(() => {
            // Получаем базовые эмбеддинги фильмов
            const baseEmbeddings = tf.gather(this.itemEmbeddingTable, itemIndices);
            
            let combinedFeatures = baseEmbeddings;
            
            // Добавляем признаки жанров если они доступны
            if (this.genreFeatures && this.numGenres > 0 && this.genreWeight) {
                const genreEmbeddings = tf.gather(this.genreFeaturesTensor, itemIndices);
                const processedGenres = tf.matMul(genreEmbeddings, this.genreWeight);
                
                // Объединяем базовые эмбеддинги с обработанными признаками жанров
                combinedFeatures = tf.add(baseEmbeddings, processedGenres);
            }
            
            // Простая нормализация
            return tf.div(combinedFeatures, tf.norm(combinedFeatures, 'euclidean', -1, true));
        });
    }
    
    // Функция скоринга: скалярное произведение между эмбеддингами пользователя и фильма
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            
            // In-batch sampled softmax loss с глубоким обучением
            const loss = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor);
                
                // Вычисляем матрицу схожести: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Метки: диагональные элементы являются позитивными парами
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            // Вычисляем градиенты и обновляем параметры
            const { value, grads } = this.optimizer.computeGradients(loss);
            
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Получаем все эмбеддинги фильмов
            const allItemIndices = tf.range(0, this.numItems, 1, 'int32');
            const allItemEmbs = this.itemForward(allItemIndices);
            
            // Вычисляем скалярное произведение с эмбеддингом пользователя
            const scores = tf.dot(allItemEmbs, userEmbedding);
            return scores.dataSync();
        });
    }
    
    getItemEmbeddings() {
        return tf.tidy(() => {
            const allItemIndices = tf.range(0, this.numItems, 1, 'int32');
            return this.itemForward(allItemIndices);
        });
    }
    
    dispose() {
        // Clean up TensorFlow.js resources
        if (this.userEmbeddingTable) {
            this.userEmbeddingTable.dispose();
        }
        if (this.itemEmbeddingTable) {
            this.itemEmbeddingTable.dispose();
        }
        if (this.genreFeaturesTensor) {
            this.genreFeaturesTensor.dispose();
        }
        if (this.userFeaturesTensor) {
            this.userFeaturesTensor.dispose();
        }
        if (this.userFeatureWeight) {
            this.userFeatureWeight.dispose();
        }
        if (this.genreWeight) {
            this.genreWeight.dispose();
        }
    }
}

// Экспорт для совместимости с существующим кодом
class TwoTowerModel extends SimpleTwoTowerModel {
    constructor(numUsers, numItems, embeddingDim) {
        super(numUsers, numItems, embeddingDim);
    }
}
