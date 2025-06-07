// MubaGPT - Application moderne optimisée
class MubaGPTApp {
    constructor() {
        // Configuration WebSocket
        this.WS_URL = `ws://${window.location.host}/ws`;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        // État de l'application
        this.currentFilters = { type: 'all' };
        this.allProperties = [];
        this.searchCriteria = {};
        this.isTyping = false;
        this.messageHistory = [];
        this.currentView = 'grid';
        
        // Éléments DOM
        this.elements = {};
        this.initializeElements();
        
        // Images génériques modernes
        this.genericImages = {
            'Appartement': 'https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Villa': 'https://images.unsplash.com/photo-1564501049412-61c2a3083791?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Maison': 'https://images.unsplash.com/photo-1572120360610-d971b9d7767c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Studio': 'https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'Bureau': 'https://images.unsplash.com/photo-1497366216548-37526070297c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            'default': 'https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
        };
        
        this.init();
    }

    // Initialisation des éléments DOM
    initializeElements() {
        this.elements = {
            chatMessages: document.getElementById('chat-messages'),
            chatForm: document.getElementById('chat-form'),
            messageInput: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            propertiesGrid: document.getElementById('properties-grid'),
            resultsCount: document.getElementById('results-count'),
            typingIndicator: document.getElementById('typing-indicator'),
            gridViewBtn: document.getElementById('grid-view'),
            listViewBtn: document.getElementById('list-view'),
            chatToggle: document.getElementById('chat-toggle'),
            connectionIndicator: document.getElementById('connection-indicator'),
            connectionText: document.getElementById('connection-text'),
            propertyCardTemplate: document.getElementById('property-card-template'),
            propertySuggestionTemplate: document.getElementById('property-suggestion-template')
        };
    }

    // Initialisation principale
    init() {
        this.initializeWebSocket();
        this.setupEventListeners();
        this.setupTextareaAutoResize();
        console.log('🚀 MubaGPT initialized successfully');
    }

    // Configuration WebSocket optimisée
    initializeWebSocket() {
        try {
            this.socket = new WebSocket(this.WS_URL);
            
            this.socket.onopen = () => {
                console.log('✅ WebSocket connection established');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
            };
            
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('❌ Error parsing WebSocket message:', error);
                }
            };
            
            this.socket.onclose = (event) => {
                console.log('❌ WebSocket connection closed:', event.code);
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };
            
            this.socket.onerror = (error) => {
                console.error('🔥 WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    // Tentative de reconnexion
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = Math.pow(2, this.reconnectAttempts) * 1000;
            this.reconnectAttempts++;
            
            console.log(`🔄 Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay/1000}s`);
            
            setTimeout(() => {
                this.initializeWebSocket();
            }, delay);
        }
    }

    // Mise à jour du statut de connexion
    updateConnectionStatus(isConnected) {
        if (this.elements.connectionIndicator && this.elements.connectionText) {
            if (isConnected) {
                this.elements.connectionIndicator.className = 'w-2 h-2 bg-emerald-500 rounded-full';
                this.elements.connectionText.textContent = 'En ligne';
            } else {
                this.elements.connectionIndicator.className = 'w-2 h-2 bg-red-500 rounded-full';
                this.elements.connectionText.textContent = 'Reconnexion...';
            }
        }
    }

    // Configuration des écouteurs d'événements
    setupEventListeners() {
        // Chat form
        this.elements.chatForm?.addEventListener('submit', (e) => this.handleChatSubmit(e));
        
        // Message input
        this.elements.messageInput?.addEventListener('input', (e) => this.handleInputChange(e));
        this.elements.messageInput?.addEventListener('keydown', (e) => this.handleKeydown(e));
        
        // Quick suggestions
        document.querySelectorAll('.quick-suggestion').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleQuickSuggestion(e));
        });
        
        // Filter chips
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.addEventListener('click', (e) => this.handleFilterChange(e));
        });
        
        // View buttons
        this.elements.gridViewBtn?.addEventListener('click', () => this.changeView('grid'));
        this.elements.listViewBtn?.addEventListener('click', () => this.changeView('list'));
        
        // Chat toggle
        this.elements.chatToggle?.addEventListener('click', () => this.toggleChatExpanded());
        
        // Window events
        window.addEventListener('resize', this.debounce(() => this.handleResize(), 300));
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    // Auto-resize du textarea
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

    // Gestion de la soumission du chat
    handleChatSubmit(event) {
        event.preventDefault();
        
        const message = this.elements.messageInput.value.trim();
        if (!message || this.isTyping) return;

        // Ajouter le message utilisateur
        this.addMessageToChat(message, 'user');
        
        // Envoyer au serveur
        this.sendMessage(message);
        
        // Afficher l'indicateur de frappe
        this.showTypingIndicator();
        
        // Effacer l'input
        this.elements.messageInput.value = '';
        this.resetTextareaHeight();
        
        // Désactiver temporairement
        this.setInputState(false);
    }

    // Gestion des changements d'input
    handleInputChange(e) {
        const message = e.target.value.trim();
        this.updateSendButtonState(message.length > 0);
    }

    // Gestion des touches du clavier
    handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleChatSubmit(e);
        }
    }

    // Gestion des suggestions rapides
    handleQuickSuggestion(e) {
        const suggestion = e.target.closest('.quick-suggestion');
        const text = suggestion.textContent.trim();
        this.elements.messageInput.value = text;
        this.elements.messageInput.focus();
        this.updateSendButtonState(true);
    }

    // Gestion des filtres
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

    // Changement de vue
    changeView(view) {
        this.currentView = view;
        
        // Mise à jour des boutons
        this.elements.gridViewBtn?.classList.toggle('active', view === 'grid');
        this.elements.listViewBtn?.classList.toggle('active', view === 'list');
        
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

    // Toggle chat expanded
    toggleChatExpanded() {
        // Fonctionnalité pour agrandir le chat en plein écran si nécessaire
        console.log('Toggle chat expanded');
    }

    // Envoi de message au serveur
    sendMessage(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            try {
                const payload = {
                    message: message,
                    sender: 'user',
                    timestamp: new Date().toISOString()
                };
                
                this.socket.send(JSON.stringify(payload));
            } catch (error) {
                console.error('Error sending message:', error);
                this.handleSendError();
            }
        } else {
            console.error('WebSocket is not connected');
            this.handleSendError();
        }
    }

    // Gestion des erreurs d'envoi
    handleSendError() {
        this.hideTypingIndicator();
        this.setInputState(true);
        this.addMessageToChat('Désolé, impossible d\'envoyer votre message. Vérifiez votre connexion.', 'bot', null, true);
    }

    // ===================================================================
    // GESTION DES MESSAGES WEBSOCKET - CORRECTION DÉFINITIVE
    // ===================================================================
    
    handleWebSocketMessage(data) {
        this.hideTypingIndicator();
        this.setInputState(true);
        
        console.log('📥 Message WebSocket reçu:', data);
        
        if (data.sender === 'bot') {
            // Ajouter le message du bot
            this.addMessageToChat(data.message, 'bot', data.properties);
            
            // ✅ FUSION CORRECTE DES PROPRIÉTÉS - SOLUTION DÉFINITIVE
            if (data.properties || data.otherProperties) {
                const mainProperties = data.properties || [];
                const otherProperties = data.otherProperties || [];
                const allProperties = [...mainProperties, ...otherProperties];
                
                console.log(`🏠 FUSION: ${mainProperties.length} principales + ${otherProperties.length} autres = ${allProperties.length} TOTAL`);
                
                // Mettre à jour le catalogue avec TOUTES les propriétés
                this.updatePropertyCatalog(allProperties);
                this.searchCriteria = data.criteria || {};
                
                // Mettre à jour le compteur
                this.updateResultsCount(allProperties.length, data.totalResults);
            }
            
            // Sauvegarder dans l'historique
            this.messageHistory.push({ 
                content: data.message, 
                sender: 'bot', 
                timestamp: Date.now() 
            });
        }
    }

    // Mettre à jour le catalogue de propriétés - SOLUTION DÉFINITIVE
    updatePropertyCatalog(properties) {
        console.log('📋 CATALOGUE: Mise à jour avec', properties?.length || 0, 'propriétés');
        
        // Sauvegarder toutes les propriétés
        this.allProperties = properties || [];
        
        // Rendre les propriétés
        this.renderProperties();
    }

    // Rendu des propriétés - LOGIQUE SIMPLE ET EFFICACE
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
        
        console.log(`🎨 RENDU: ${filteredProperties.length} propriétés à afficher`);
        
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
                
                console.log(`✅ Propriété ${index + 1} ajoutée: ${property.neighborhood || 'N/A'}`);
            }
        });
        
        console.log(`🏁 RENDU TERMINÉ: ${grid.children.length} cartes dans le DOM`);
    }

    // Créer une carte de propriété
    createPropertyCard(property) {
        if (!this.elements.propertyCardTemplate) {
            console.error('❌ Template propertyCardTemplate non trouvé');
            return null;
        }
        
        const clone = this.elements.propertyCardTemplate.content.cloneNode(true);
        
        // Sélectionner les éléments
        const imageElement = clone.querySelector('.property-image');
        const typeElement = clone.querySelector('.property-type');
        const neighborhoodElement = clone.querySelector('.property-neighborhood');
        const priceElement = clone.querySelector('.property-price');
        const bedroomsElements = clone.querySelectorAll('.property-bedrooms');
        const bathroomsElements = clone.querySelectorAll('.property-bathrooms');
        const areaElements = clone.querySelectorAll('.property-area');
        const urlElement = clone.querySelector('.property-url');
        
        // Image avec fallback
        const propertyType = property.property_type || 'default';
        const imageUrl = this.genericImages[propertyType] || this.genericImages['default'];
        
        if (imageElement) {
            imageElement.src = imageUrl;
            imageElement.alt = `${propertyType} à ${property.neighborhood}`;
            imageElement.onerror = () => {
                imageElement.src = this.genericImages['default'];
            };
        }
        
        // Remplir les informations
        if (typeElement) typeElement.textContent = propertyType;
        if (neighborhoodElement) neighborhoodElement.textContent = property.neighborhood || 'N/A';
        if (priceElement) priceElement.textContent = this.formatPrice(property.price);
        
        bedroomsElements.forEach(el => el.textContent = property.bedrooms || 'N/A');
        bathroomsElements.forEach(el => el.textContent = property.bathrooms || 'N/A');
        areaElements.forEach(el => el.textContent = property.area || 'N/A');
        
        // URL avec target blank
        if (urlElement && property.url) {
            urlElement.href = property.url;
            urlElement.target = '_blank';
            urlElement.rel = 'noopener noreferrer';
        }
        
        return clone;
    }

    // Obtenir les propriétés filtrées
    getFilteredProperties() {
        let filtered = [...this.allProperties];
        
        if (this.currentFilters.type && this.currentFilters.type !== 'all') {
            const filterType = this.currentFilters.type.toLowerCase();
            filtered = filtered.filter(prop => {
                const propType = (prop.property_type || '').toLowerCase();
                return propType.includes(filterType) || filterType.includes(propType);
            });
        }
        
        console.log(`🔍 FILTRE: ${this.allProperties.length} → ${filtered.length} propriétés (filtre: ${this.currentFilters.type})`);
        return filtered;
    }

    // Affichage de l'état vide
    showEmptyState() {
        const grid = this.elements.propertiesGrid;
        if (!grid) return;
        
        grid.innerHTML = `
            <div class="col-span-full flex flex-col items-center justify-center py-24 text-center">
                <div class="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-2xl flex items-center justify-center mb-6">
                    <i class="fas fa-search-minus text-blue-600 text-2xl"></i>
                </div>
                <h3 class="text-xl font-semibold text-gray-800 mb-3">Aucune propriété trouvée</h3>
                <p class="text-gray-500 max-w-md leading-relaxed">Essayez de modifier vos critères de recherche ou démarrez une nouvelle conversation.</p>
                <button onclick="window.mubaGPT.resetFilters()" class="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    Réinitialiser les filtres
                </button>
            </div>
        `;
    }

    // Réinitialiser les filtres
    resetFilters() {
        this.currentFilters = { type: 'all' };
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.classList.toggle('active', chip.dataset.filter === 'all');
        });
        this.renderProperties();
    }

    // Filtrer les propriétés
    filterProperties() {
        this.renderProperties();
    }

    // Mettre à jour le compteur de résultats
    updateResultsCount(displayed, total) {
        if (this.elements.resultsCount) {
            if (displayed === 0) {
                this.elements.resultsCount.textContent = 'Aucun résultat trouvé';
            } else if (displayed === total) {
                this.elements.resultsCount.textContent = `${displayed} propriété${displayed > 1 ? 's' : ''} trouvée${displayed > 1 ? 's' : ''}`;
            } else {
                this.elements.resultsCount.textContent = `${displayed} sur ${total} propriétés affichées`;
            }
        }
    }

    // Ajouter un message au chat
    addMessageToChat(message, sender, properties = null, isError = false) {
        const messagesContainer = this.elements.chatMessages.querySelector('.space-y-6') || this.elements.chatMessages;
        
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

    // Créer un message utilisateur
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

    // Créer un message bot
    createBotMessage(content, properties = null, isError = false) {
        const errorClass = isError ? 'border-red-300 bg-red-50 text-red-700' : '';
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

    // Créer HTML pour suggestion de propriété
    createPropertySuggestionHTML(property) {
        const propertyType = property.property_type || 'default';
        const imageUrl = this.genericImages[propertyType] || this.genericImages['default'];
        
        return `
            <div class="property-suggestion bg-white border border-gray-100 rounded-xl p-4 hover:shadow-lg hover:border-blue-200 transition-all duration-300 cursor-pointer" 
                 onclick="window.open('${property.url || '#'}', '_blank')">
                <div class="flex items-center space-x-4">
                    <div class="w-16 h-16 bg-gray-100 rounded-xl overflow-hidden flex-shrink-0">
                        <img class="w-full h-full object-cover" src="${imageUrl}" alt="${propertyType}" onerror="this.src='${this.genericImages['default']}'">
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

    // Indicateurs de frappe
    showTypingIndicator() {
        this.isTyping = true;
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.classList.remove('hidden');
        }
    }

    hideTypingIndicator() {
        this.isTyping = false;
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.classList.add('hidden');
        }
    }

    // Utilitaires
    formatPrice(price) {
        if (!price) return 'Prix non disponible';
        if (typeof price === 'string') return price;
        
        return new Intl.NumberFormat('fr-MA', {
            style: 'currency',
            currency: 'MAD',
            maximumFractionDigits: 0
        }).format(price);
    }

    formatTime(date) {
        return date.toLocaleTimeString('fr-FR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

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

    // Gestion des états
    setInputState(enabled) {
        if (this.elements.messageInput) {
            this.elements.messageInput.disabled = !enabled;
        }
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = !enabled;
        }
    }

    updateSendButtonState(hasContent) {
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = !hasContent || this.isTyping;
        }
    }

    resetTextareaHeight() {
        if (this.elements.messageInput) {
            this.elements.messageInput.style.height = '60px';
        }
    }

    scrollToBottom() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTo({
                top: this.elements.chatMessages.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    handleResize() {
        // Logique de redimensionnement si nécessaire
    }

    // Nettoyage
    cleanup() {
        if (this.socket) {
            this.socket.close();
        }
    }

    // Fonction de test pour déboguer
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
}

// Initialisation de l'application
document.addEventListener('DOMContentLoaded', () => {
    window.mubaGPT = new MubaGPTApp();
    console.log('🎉 MubaGPT ready!');
});