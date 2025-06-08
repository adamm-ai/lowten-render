// ========================================================================================
// MUBAGPT - APPLICATION JAVASCRIPT COMPLÈTE ET OPTIMISÉE POUR RENDER
// Version production-ready avec gestion d'erreurs robuste
// ========================================================================================

class MubaGPTApp {
    constructor() {
        console.log('🚀 Initialisation MubaGPT...');
        
        // Configuration WebSocket - RENDER COMPATIBLE
        this.WS_URL = this._getWebSocketURL();
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        // État de l'application
        this.currentFilters = { type: 'all' };
        this.allProperties = [];
        this.searchCriteria = {};
        this.isTyping = false;
        this.messageHistory = [];
        this.currentView = 'grid';
        this.connectionStatus = 'disconnected';
        
        // Éléments DOM
        this.elements = {};
        
        // Images génériques modernes (Unsplash)
        this.genericImages = {
            'Appartement': 'https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Villa': 'https://images.unsplash.com/photo-1564501049412-61c2a3083791?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Maison': 'https://images.unsplash.com/photo-1572120360610-d971b9d7767c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Studio': 'https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Bureau': 'https://images.unsplash.com/photo-1497366216548-37526070297c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'default': 'https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
        };
        
        // Initialisation
        this.init();
    }

    // ========================================================================================
    // INITIALISATION ET CONFIGURATION
    // ========================================================================================

    /**
     * Détermine l'URL WebSocket en fonction de l'environnement
     */
    _getWebSocketURL() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        
        // Pour Render et production
        if (host.includes('.onrender.com') || host.includes('render.com')) {
            return `wss://${host}/ws`;
        }
        
        // Pour développement local
        if (host.includes('localhost') || host.includes('127.0.0.1')) {
            return `ws://${host}/ws`;
        }
        
        // Fallback
        return `${protocol}//${host}/ws`;
    }

    /**
     * Initialisation des éléments DOM avec vérification de sécurité
     */
    initializeElements() {
        const elementIds = [
            'chat-messages', 'chat-form', 'message-input', 'send-button',
            'properties-grid', 'results-count', 'typing-indicator',
            'grid-view', 'list-view', 'chat-toggle', 'connection-indicator',
            'connection-text', 'property-card-template', 'property-suggestion-template',
            'char-count', 'clear-chat'
        ];

        elementIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                this.elements[this._camelCase(id)] = element;
                console.log(`✅ Élément trouvé: ${id}`);
            } else {
                console.warn(`⚠️ Élément manquant: ${id}`);
            }
        });

        // Vérifier les éléments critiques
        const criticalElements = ['chatMessages', 'chatForm', 'messageInput', 'propertiesGrid'];
        const missingCritical = criticalElements.filter(key => !this.elements[key]);
        
        if (missingCritical.length > 0) {
            console.error('❌ Éléments critiques manquants:', missingCritical);
            this.showErrorMessage('Erreur d\'initialisation. Veuillez recharger la page.');
            return false;
        }

        console.log('✅ Tous les éléments critiques trouvés');
        return true;
    }

    /**
     * Convertit kebab-case en camelCase
     */
    _camelCase(str) {
        return str.replace(/-([a-z])/g, (match, letter) => letter.toUpperCase());
    }

    /**
     * Initialisation principale
     */
    async init() {
        try {
            console.log('🔧 Initialisation des éléments DOM...');
            if (!this.initializeElements()) {
                return;
            }

            console.log('🎛️ Configuration des événements...');
            this.setupEventListeners();
            
            console.log('📐 Configuration du textarea auto-resize...');
            this.setupTextareaAutoResize();
            
            console.log('🌐 Tentative de connexion WebSocket...');
            await this.initializeWebSocket();
            
            console.log('🎉 MubaGPT initialisé avec succès !');
            this.updateConnectionStatus('ready');
            
        } catch (error) {
            console.error('❌ Erreur lors de l\'initialisation:', error);
            this.showErrorMessage('Erreur d\'initialisation. Certaines fonctionnalités peuvent ne pas fonctionner.');
        }
    }

    // ========================================================================================
    // WEBSOCKET ET COMMUNICATION
    // ========================================================================================

    /**
     * Initialisation WebSocket avec gestion d'erreurs robuste
     */
    async initializeWebSocket() {
        try {
            console.log(`🔌 Connexion WebSocket: ${this.WS_URL}`);
            
            this.socket = new WebSocket(this.WS_URL);
            
            this.socket.onopen = () => {
                console.log('✅ WebSocket connecté');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                this.showSystemMessage('Connexion établie avec l\'assistant IA');
            };
            
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('❌ Erreur parsing message WebSocket:', error);
                    this.showErrorMessage('Erreur de communication avec le serveur');
                }
            };
            
            this.socket.onclose = (event) => {
                console.log(`❌ WebSocket fermé (code: ${event.code})`);
                this.updateConnectionStatus('disconnected');
                this.attemptReconnect();
            };
            
            this.socket.onerror = (error) => {
                console.error('🔥 Erreur WebSocket:', error);
                this.updateConnectionStatus('error');
            };
            
        } catch (error) {
            console.error('❌ Impossible d\'initialiser WebSocket:', error);
            this.updateConnectionStatus('error');
            this.showErrorMessage('Impossible de se connecter au serveur. Mode déconnecté activé.');
        }
    }

    /**
     * Tentative de reconnexion automatique
     */
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
            this.reconnectAttempts++;
            
            console.log(`🔄 Reconnexion ${this.reconnectAttempts}/${this.maxReconnectAttempts} dans ${delay/1000}s`);
            this.updateConnectionStatus('reconnecting');
            
            setTimeout(() => {
                this.initializeWebSocket();
            }, delay);
        } else {
            console.error('❌ Échec de reconnexion après', this.maxReconnectAttempts, 'tentatives');
            this.updateConnectionStatus('failed');
            this.showErrorMessage('Connexion perdue. Veuillez recharger la page.');
        }
    }

    /**
     * Mise à jour du statut de connexion UI
     */
    updateConnectionStatus(status) {
        this.connectionStatus = status;
        
        const indicator = this.elements.connectionIndicator;
        const text = this.elements.connectionText;
        
        if (!indicator || !text) return;

        const statusConfig = {
            'connected': {
                color: 'bg-emerald-500',
                text: 'En ligne',
                pulse: false
            },
            'disconnected': {
                color: 'bg-red-500', 
                text: 'Déconnecté',
                pulse: true
            },
            'reconnecting': {
                color: 'bg-yellow-500',
                text: 'Reconnexion...',
                pulse: true
            },
            'error': {
                color: 'bg-red-600',
                text: 'Erreur',
                pulse: true
            },
            'ready': {
                color: 'bg-blue-500',
                text: 'Prêt',
                pulse: false
            },
            'failed': {
                color: 'bg-gray-500',
                text: 'Hors ligne',
                pulse: false
            }
        };

        const config = statusConfig[status] || statusConfig['disconnected'];
        
        indicator.className = `w-2 h-2 ${config.color} rounded-full ${config.pulse ? 'animate-pulse' : ''}`;
        text.textContent = config.text;
    }

    // ========================================================================================
    // GESTION DES ÉVÉNEMENTS
    // ========================================================================================

    /**
     * Configuration des écouteurs d'événements
     */
    setupEventListeners() {
        try {
            // Formulaire de chat
            if (this.elements.chatForm) {
                this.elements.chatForm.addEventListener('submit', (e) => this.handleChatSubmit(e));
            }

            // Input de message
            if (this.elements.messageInput) {
                this.elements.messageInput.addEventListener('input', (e) => this.handleInputChange(e));
                this.elements.messageInput.addEventListener('keydown', (e) => this.handleKeydown(e));
            }

            // Suggestions rapides
            document.querySelectorAll('.quick-suggestion').forEach(btn => {
                btn.addEventListener('click', (e) => this.handleQuickSuggestion(e));
            });

            // Filtres
            document.querySelectorAll('.filter-chip').forEach(chip => {
                chip.addEventListener('click', (e) => this.handleFilterChange(e));
            });

            // Boutons de vue
            if (this.elements.gridView) {
                this.elements.gridView.addEventListener('click', () => this.changeView('grid'));
            }
            if (this.elements.listView) {
                this.elements.listView.addEventListener('click', () => this.changeView('list'));
            }

            // Bouton toggle chat
            if (this.elements.chatToggle) {
                this.elements.chatToggle.addEventListener('click', () => this.toggleChatExpanded());
            }

            // Bouton clear chat
            if (this.elements.clearChat) {
                this.elements.clearChat.addEventListener('click', () => this.clearChat());
            }

            // Événements globaux
            window.addEventListener('resize', this.debounce(() => this.handleResize(), 300));
            window.addEventListener('beforeunload', () => this.cleanup());
            
            // Gestion des erreurs globales
            window.addEventListener('error', (e) => {
                console.error('❌ Erreur globale:', e.error);
            });

            console.log('✅ Tous les événements configurés');

        } catch (error) {
            console.error('❌ Erreur configuration événements:', error);
        }
    }

    /**
     * Auto-resize du textarea
     */
    setupTextareaAutoResize() {
        const textarea = this.elements.messageInput;
        if (!textarea) return;

        const autoResize = () => {
            textarea.style.height = 'auto';
            const newHeight = Math.min(textarea.scrollHeight, 128);
            textarea.style.height = newHeight + 'px';
        };

        textarea.addEventListener('input', autoResize);
        autoResize();
    }

    // ========================================================================================
    // GESTION DES INTERACTIONS UTILISATEUR
    // ========================================================================================

    /**
     * Gestion de la soumission du formulaire de chat
     */
    async handleChatSubmit(event) {
        event.preventDefault();
        
        const message = this.elements.messageInput?.value?.trim();
        if (!message || this.isTyping) return;

        try {
            // Ajouter le message utilisateur immédiatement
            this.addMessageToChat(message, 'user');
            
            // Envoyer au serveur
            await this.sendMessage(message);
            
            // Afficher l'indicateur de frappe
            this.showTypingIndicator();
            
            // Nettoyer l'input
            this.elements.messageInput.value = '';
            this.resetTextareaHeight();
            this.updateCharCount(0);
            
            // Désactiver temporairement
            this.setInputState(false);

        } catch (error) {
            console.error('❌ Erreur soumission chat:', error);
            this.showErrorMessage('Erreur lors de l\'envoi du message');
            this.hideTypingIndicator();
            this.setInputState(true);
        }
    }

    /**
     * Gestion des changements dans l'input
     */
    handleInputChange(e) {
        const message = e.target.value.trim();
        const length = e.target.value.length;
        
        this.updateSendButtonState(message.length > 0);
        this.updateCharCount(length);
    }

    /**
     * Gestion des touches du clavier
     */
    handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleChatSubmit(e);
        }
    }

    /**
     * Gestion des suggestions rapides
     */
    handleQuickSuggestion(e) {
        const suggestion = e.target.closest('.quick-suggestion');
        const text = suggestion.dataset.text || suggestion.textContent.trim();
        
        if (this.elements.messageInput) {
            this.elements.messageInput.value = text;
            this.elements.messageInput.focus();
            this.updateSendButtonState(true);
            this.updateCharCount(text.length);
        }
    }

    /**
     * Gestion des filtres
     */
    handleFilterChange(e) {
        const chip = e.target.closest('.filter-chip');
        const filter = chip.dataset.filter;
        
        // Mise à jour visuelle
        document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        
        // Appliquer le filtre
        this.currentFilters.type = filter;
        this.filterProperties();
    }

    /**
     * Changement de vue (grille/liste)
     */
    changeView(view) {
        this.currentView = view;
        
        // Mise à jour des boutons
        if (this.elements.gridView) {
            this.elements.gridView.classList.toggle('active', view === 'grid');
        }
        if (this.elements.listView) {
            this.elements.listView.classList.toggle('active', view === 'list');
        }
        
        // Mise à jour de la grille
        const grid = this.elements.propertiesGrid;
        if (grid) {
            if (view === 'list') {
                grid.className = 'p-6 space-y-4';
            } else {
                grid.className = 'p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6';
            }
        }
        
        // Re-rendre les propriétés
        if (this.allProperties.length > 0) {
            this.renderProperties();
        }
    }

    /**
     * Toggle chat en mode étendu
     */
    toggleChatExpanded() {
        console.log('🔄 Toggle chat étendu (à implémenter)');
        // TODO: Implémenter mode plein écran pour mobile
    }

    /**
     * Vider le chat
     */
    clearChat() {
        if (confirm('Voulez-vous vraiment vider la conversation ?')) {
            const messagesContainer = this.elements.chatMessages?.querySelector('.space-y-6');
            if (messagesContainer) {
                // Garder seulement le message de bienvenue
                const welcomeMessage = messagesContainer.querySelector('.message-wrapper.bot');
                messagesContainer.innerHTML = '';
                if (welcomeMessage) {
                    messagesContainer.appendChild(welcomeMessage);
                }
            }
            
            this.messageHistory = [];
            this.allProperties = [];
            this.renderProperties();
            console.log('🧹 Chat vidé');
        }
    }

    // ========================================================================================
    // COMMUNICATION AVEC LE SERVEUR
    // ========================================================================================

    /**
     * Envoi de message au serveur
     */
    async sendMessage(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            try {
                const payload = {
                    message: message,
                    sender: 'user',
                    timestamp: new Date().toISOString(),
                    client_id: this.getClientId()
                };
                
                this.socket.send(JSON.stringify(payload));
                console.log('📤 Message envoyé:', message.substring(0, 50) + '...');
                
            } catch (error) {
                console.error('❌ Erreur envoi message:', error);
                throw error;
            }
        } else {
            console.error('❌ WebSocket non connecté');
            throw new Error('Connexion non disponible');
        }
    }

    /**
     * Génère ou récupère un ID client unique
     */
    getClientId() {
        let clientId = localStorage.getItem('mubagpt_client_id');
        if (!clientId) {
            clientId = 'client_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('mubagpt_client_id', clientId);
        }
        return clientId;
    }

    /**
     * Gestion des messages WebSocket reçus
     */
    handleWebSocketMessage(data) {
        try {
            this.hideTypingIndicator();
            this.setInputState(true);
            
            console.log('📥 Message WebSocket reçu:', data);
            
            if (data.sender === 'bot') {
                // Ajouter le message du bot
                this.addMessageToChat(data.message, 'bot', data.properties);
                
                // Gérer les propriétés
                if (data.properties || data.otherProperties) {
                    const mainProperties = data.properties || [];
                    const otherProperties = data.otherProperties || [];
                    const allProperties = [...mainProperties, ...otherProperties];
                    
                    console.log(`🏠 Propriétés reçues: ${allProperties.length} total`);
                    
                    this.updatePropertyCatalog(allProperties);
                    this.searchCriteria = data.criteria || {};
                    this.updateResultsCount(allProperties.length, data.totalResults);
                }
                
                // Sauvegarder dans l'historique
                this.messageHistory.push({ 
                    content: data.message, 
                    sender: 'bot', 
                    timestamp: Date.now() 
                });
            }
            
        } catch (error) {
            console.error('❌ Erreur traitement message WebSocket:', error);
            this.showErrorMessage('Erreur lors du traitement de la réponse');
        }
    }

    // ========================================================================================
    // GESTION DES PROPRIÉTÉS
    // ========================================================================================

    /**
     * Met à jour le catalogue de propriétés
     */
    updatePropertyCatalog(properties) {
        console.log('📋 Mise à jour catalogue:', properties?.length || 0, 'propriétés');
        
        this.allProperties = properties || [];
        this.renderProperties();
    }

    /**
     * Rendu des propriétés
     */
    renderProperties() {
        const grid = this.elements.propertiesGrid;
        if (!grid) {
            console.error('❌ Élément propertiesGrid non trouvé');
            return;
        }
        
        // Effacer le contenu actuel
        grid.innerHTML = '';
        
        // Obtenir les propriétés filtrées
        const filteredProperties = this.getFilteredProperties();
        
        console.log(`🎨 Rendu: ${filteredProperties.length} propriétés à afficher`);
        
        if (filteredProperties.length === 0) {
            this.showEmptyState();
            return;
        }
        
        // Ajouter chaque propriété avec animation
        filteredProperties.forEach((property, index) => {
            const card = this.createPropertyCard(property);
            if (card) {
                // Animation d'entrée
                const cardElement = card.firstElementChild || card;
                cardElement.style.opacity = '0';
                cardElement.style.transform = 'translateY(20px)';
                
                grid.appendChild(card);
                
                // Animation avec délai
                setTimeout(() => {
                    cardElement.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                    cardElement.style.opacity = '1';
                    cardElement.style.transform = 'translateY(0)';
                }, index * 100);
                
                console.log(`✅ Propriété ${index + 1} ajoutée`);
            }
        });
        
        console.log(`🏁 Rendu terminé: ${grid.children.length} cartes`);
    }

    /**
     * Crée une carte de propriété
     */
    createPropertyCard(property) {
        if (!this.elements.propertyCardTemplate) {
            console.error('❌ Template propertyCardTemplate non trouvé');
            return null;
        }
        
        try {
            const clone = this.elements.propertyCardTemplate.content.cloneNode(true);
            
            // Sélectionner les éléments
            const imageElement = clone.querySelector('.property-image');
            const typeElement = clone.querySelector('.property-type');
            const neighborhoodElement = clone.querySelector('.property-neighborhood');
            const priceElements = clone.querySelectorAll('.property-price, .property-price-badge');
            const bedroomsElements = clone.querySelectorAll('.property-bedrooms');
            const bathroomsElements = clone.querySelectorAll('.property-bathrooms');
            const areaElements = clone.querySelectorAll('.property-area');
            const urlElement = clone.querySelector('.property-url');
            
            // Image avec fallback
            const propertyType = property.property_type || 'default';
            const imageUrl = this.genericImages[propertyType] || this.genericImages['default'];
            
            if (imageElement) {
                imageElement.src = imageUrl;
                imageElement.alt = `${propertyType} à ${property.neighborhood || 'N/A'}`;
                imageElement.onerror = () => {
                    imageElement.src = this.genericImages['default'];
                };
            }
            
            // Remplir les informations
            if (typeElement) typeElement.textContent = propertyType;
            if (neighborhoodElement) neighborhoodElement.textContent = property.neighborhood || 'Localisation non spécifiée';
            
            const formattedPrice = this.formatPrice(property.price);
            priceElements.forEach(el => el.textContent = formattedPrice);
            
            bedroomsElements.forEach(el => el.textContent = property.bedrooms || 'N/A');
            bathroomsElements.forEach(el => el.textContent = property.bathrooms || 'N/A');
            areaElements.forEach(el => el.textContent = property.area || 'N/A');
            
            // URL avec sécurité
            if (urlElement && property.url) {
                urlElement.href = property.url;
                urlElement.target = '_blank';
                urlElement.rel = 'noopener noreferrer';
            }
            
            return clone;
            
        } catch (error) {
            console.error('❌ Erreur création carte propriété:', error);
            return null;
        }
    }

    /**
     * Obtient les propriétés filtrées
     */
    getFilteredProperties() {
        let filtered = [...this.allProperties];
        
        if (this.currentFilters.type && this.currentFilters.type !== 'all') {
            const filterType = this.currentFilters.type.toLowerCase();
            filtered = filtered.filter(prop => {
                const propType = (prop.property_type || '').toLowerCase();
                return propType.includes(filterType) || filterType.includes(propType);
            });
        }
        
        console.log(`🔍 Filtre: ${this.allProperties.length} → ${filtered.length} propriétés`);
        return filtered;
    }

    /**
     * Affiche l'état vide
     */
    showEmptyState() {
        const grid = this.elements.propertiesGrid;
        if (!grid) return;
        
        grid.innerHTML = `
            <div class="col-span-full flex flex-col items-center justify-center py-24 text-center">
                <div class="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-2xl flex items-center justify-center mb-6">
                    <i class="fas fa-search-minus text-blue-600 text-2xl"></i>
                </div>
                <h3 class="text-xl font-semibold text-gray-800 mb-3">Aucune propriété trouvée</h3>
                <p class="text-gray-500 max-w-md leading-relaxed mb-4">
                    Aucune propriété ne correspond à vos critères actuels. Essayez de modifier vos filtres.
                </p>
                <button onclick="window.mubaGPT.resetFilters()" class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    Réinitialiser les filtres
                </button>
            </div>
        `;
    }

    /**
     * Réinitialise les filtres
     */
    resetFilters() {
        this.currentFilters = { type: 'all' };
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.classList.toggle('active', chip.dataset.filter === 'all');
        });
        this.renderProperties();
        console.log('🔄 Filtres réinitialisés');
    }

    /**
     * Filtre les propriétés
     */
    filterProperties() {
        this.renderProperties();
    }

    /**
     * Met à jour le compteur de résultats
     */
    updateResultsCount(displayed, total) {
        if (this.elements.resultsCount) {
            if (displayed === 0) {
                this.elements.resultsCount.textContent = 'Aucun résultat trouvé';
            } else if (displayed === total || !total) {
                this.elements.resultsCount.textContent = `${displayed} propriété${displayed > 1 ? 's' : ''} trouvée${displayed > 1 ? 's' : ''}`;
            } else {
                this.elements.resultsCount.textContent = `${displayed} sur ${total} propriétés affichées`;
            }
        }
    }

    // ========================================================================================
    // GESTION DES MESSAGES
    // ========================================================================================

    /**
     * Ajoute un message au chat
     */
    addMessageToChat(message, sender, properties = null, isError = false) {
        const messagesContainer = this.elements.chatMessages?.querySelector('.space-y-6');
        if (!messagesContainer) {
            console.error('❌ Container de messages non trouvé');
            return;
        }
        
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message-wrapper ${sender}`;
        
        const messageHTML = sender === 'user' 
            ? this.createUserMessage(message)
            : this.createBotMessage(message, properties, isError);
        
        messageWrapper.innerHTML = messageHTML;
        messagesContainer.appendChild(messageWrapper);
        
        // Scroll vers le bas
        this.scrollToBottom();
    }

    /**
     * Crée un message utilisateur
     */
    createUserMessage(content) {
        return `
            <div class="flex items-start space-x-4 justify-end">
                <div class="flex-1">
                    <div class="message-content user">
                        <p>${this.escapeHtml(content)}</p>
                    </div>
                    <div class="message-time text-right">${this.formatTime(new Date())}</div>
                </div>
                <div class="w-9 h-9 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-user text-white text-sm"></i>
                </div>
            </div>
        `;
    }

    /**
     * Crée un message bot
     */
    createBotMessage(content, properties = null, isError = false) {
        const errorClass = isError ? 'error' : '';
        let propertiesHTML = '';
        
        if (properties && properties.length > 0) {
            propertiesHTML = `
                <div class="mt-4 space-y-3">
                    ${properties.slice(0, 3).map(property => this.createPropertySuggestionHTML(property)).join('')}
                </div>
            `;
        }
        
        return `
            <div class="flex items-start space-x-4">
                <div class="w-9 h-9 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-robot text-white text-sm"></i>
                </div>
                <div class="flex-1">
                    <div class="message-content bot ${errorClass}">
                        <p>${this.escapeHtml(content)}</p>
                        ${propertiesHTML}
                    </div>
                    <div class="message-time">${this.formatTime(new Date())}</div>
                </div>
            </div>
        `;
    }

    /**
     * Crée HTML pour suggestion de propriété
     */
    createPropertySuggestionHTML(property) {
        const propertyType = property.property_type || 'default';
        const imageUrl = this.genericImages[propertyType] || this.genericImages['default'];
        
        return `
            <div class="property-suggestion bg-white border border-gray-100 rounded-xl p-4 hover:shadow-lg hover:border-blue-200 transition-all duration-300 cursor-pointer" 
                 onclick="window.open('${property.url || '#'}', '_blank')">
                <div class="flex items-center space-x-4">
                    <div class="w-16 h-16 bg-gray-100 rounded-xl overflow-hidden flex-shrink-0">
                        <img class="w-full h-full object-cover" src="${imageUrl}" alt="${propertyType}" 
                             onerror="this.src='${this.genericImages['default']}'">
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="flex justify-between items-start mb-2">
                            <div class="font-semibold text-gray-900 text-sm truncate">${property.neighborhood || 'N/A'}</div>
                            <div class="text-blue-600 font-bold text-sm ml-3">${this.formatPrice(property.price)}</div>
                        </div>
                        <div class="text-xs text-gray-500 flex space-x-3">
                            <span><i class="fas fa-bed mr-1"></i>${property.bedrooms || 'N/A'}</span>
                            <span><i class="fas fa-bath mr-1"></i>${property.bathrooms || 'N/A'}</span>
                            <span><i class="fas fa-ruler-combined mr-1"></i>${property.area || 'N/A'} m²</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Affiche un message système
     */
    showSystemMessage(message) {
        this.addMessageToChat(`ℹ️ ${message}`, 'system');
    }

    /**
     * Affiche un message d'erreur
     */
    showErrorMessage(message) {
        this.addMessageToChat(`❌ ${message}`, 'bot', null, true);
    }

    // ========================================================================================
    // INDICATEURS ET ÉTATS
    // ========================================================================================

    /**
     * Affiche l'indicateur de frappe
     */
    showTypingIndicator() {
        this.isTyping = true;
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.classList.remove('hidden');
        }
    }

    /**
     * Cache l'indicateur de frappe
     */
    hideTypingIndicator() {
        this.isTyping = false;
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.classList.add('hidden');
        }
    }

    /**
     * Définit l'état des inputs
     */
    setInputState(enabled) {
        if (this.elements.messageInput) {
            this.elements.messageInput.disabled = !enabled;
        }
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = !enabled || this.isTyping;
        }
    }

    /**
     * Met à jour l'état du bouton d'envoi
     */
    updateSendButtonState(hasContent) {
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = !hasContent || this.isTyping;
        }
    }

    /**
     * Met à jour le compteur de caractères
     */
    updateCharCount(count) {
        if (this.elements.charCount) {
            this.elements.charCount.textContent = count;
        }
    }

    /**
     * Remet à zéro la hauteur du textarea
     */
    resetTextareaHeight() {
        if (this.elements.messageInput) {
            this.elements.messageInput.style.height = '60px';
        }
    }

    /**
     * Scroll vers le bas du chat
     */
    scrollToBottom() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTo({
                top: this.elements.chatMessages.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    // ========================================================================================
    // UTILITAIRES
    // ========================================================================================

    /**
     * Formate le prix
     */
    formatPrice(price) {
        if (!price) return 'Prix non disponible';
        if (typeof price === 'string') return price;
        
        try {
            return new Intl.NumberFormat('fr-MA', {
                style: 'currency',
                currency: 'MAD',
                maximumFractionDigits: 0
            }).format(price);
        } catch (error) {
            return `${price.toLocaleString()} MAD`;
        }
    }

    /**
     * Formate l'heure
     */
    formatTime(date) {
        return date.toLocaleTimeString('fr-FR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    /**
     * Échappe le HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Gestion du redimensionnement
     */
    handleResize() {
        // Ajustements responsive si nécessaire
        console.log('🔄 Fenêtre redimensionnée');
    }

    /**
     * Nettoyage avant fermeture
     */
    cleanup() {
        if (this.socket) {
            this.socket.close();
            console.log('🧹 WebSocket fermé');
        }
    }

    // ========================================================================================
    // MÉTHODES DE TEST ET DEBUG
    // ========================================================================================

    /**
     * Crée des propriétés de test
     */
    createTestProperties() {
        const testProperties = [
            {
                neighborhood: "Casablanca Finance City",
                price: 3340000,
                bedrooms: 3,
                bathrooms: 2,
                area: 158,
                property_type: "Appartement",
                url: "https://example.com/property1"
            },
            {
                neighborhood: "Hay Riad",
                price: 2500000,
                bedrooms: 3,
                bathrooms: 2,
                area: 120,
                property_type: "Villa",
                url: "https://example.com/property2"
            },
            {
                neighborhood: "Guéliz",
                price: 850000,
                bedrooms: 1,
                bathrooms: 1,
                area: 60,
                property_type: "Studio",
                url: "https://example.com/property3"
            }
        ];
        
        console.log('🧪 Test avec', testProperties.length, 'propriétés factices');
        this.updatePropertyCatalog(testProperties);
        return testProperties;
    }

    /**
     * Informations de debug
     */
    getDebugInfo() {
        return {
            connectionStatus: this.connectionStatus,
            propertiesCount: this.allProperties.length,
            messageHistoryCount: this.messageHistory.length,
            currentFilters: this.currentFilters,
            currentView: this.currentView,
            isTyping: this.isTyping,
            elementsFound: Object.keys(this.elements),
            websocketUrl: this.WS_URL,
            clientId: this.getClientId()
        };
    }
}

// ========================================================================================
// INITIALISATION GLOBALE
// ========================================================================================

// Initialisation when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('🌟 DOM chargé, initialisation MubaGPT...');
        
        // Créer l'instance globale
        window.mubaGPT = new MubaGPTApp();
        
        // Debug info
        console.log('🎉 MubaGPT initialisé avec succès !');
        console.log('🔍 Info debug:', window.mubaGPT.getDebugInfo());
        
        // Test automatique si en développement
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            console.log('🧪 Mode développement détecté');
            // window.mubaGPT.createTestProperties(); // Décommentez pour tester
        }
        
    } catch (error) {
        console.error('❌ Erreur fatale lors de l\'initialisation:', error);
        
        // Afficher un message d'erreur à l'utilisateur
        document.body.insertAdjacentHTML('beforeend', `
            <div style="position: fixed; top: 20px; right: 20px; background: #fee; border: 1px solid #fcc; color: #c66; padding: 15px; border-radius: 8px; z-index: 9999; font-family: sans-serif; max-width: 300px;">
                <strong>Erreur d'initialisation</strong><br>
                L'application n'a pas pu démarrer correctement. Veuillez recharger la page.
                <button onclick="location.reload()" style="margin-top: 10px; padding: 5px 10px; background: #c66; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Recharger
                </button>
            </div>
        `);
    }
});

// Gestion des erreurs globales
window.addEventListener('error', (e) => {
    console.error('❌ Erreur JavaScript globale:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('❌ Promise rejetée non gérée:', e.reason);
});

console.log('📄 Script MubaGPT chargé');