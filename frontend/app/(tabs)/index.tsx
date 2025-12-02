import React, { useEffect, useRef, useState } from 'react';
// Imports necessary for a standard React Native / Expo application
import { FontAwesome, MaterialIcons } from '@expo/vector-icons';
import axios from 'axios';
import * as ImagePicker from 'expo-image-picker';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
  KeyboardAvoidingView, Platform,
  ScrollView,
  StyleSheet, Text,
  TextInput,
  TouchableOpacity,
  View
} from 'react-native';

// ==================================================================
// ⚙️ CONFIGURATION FOR THE USER
// ==================================================================

// 1. FIND YOUR LOCAL IP ADDRESS:
//    - Windows: Open CMD, type 'ipconfig', look for "IPv4 Address"
//    - Mac/Linux: Open Terminal, type 'ifconfig' or 'ip a'
// 2. PASTE IT BELOW:
const SERVER_IP = '10.227.24.201'; // <--- CHANGE THIS to your computer's IP
const PORT = '5000';

const API_BASE_URL = `http://${SERVER_IP}:${PORT}`; 

// ==================================================================


const { width, height } = Dimensions.get('window');

// Define the type for the classification result data
interface PlantResult {
    name: string;
    is_medicinal: boolean;
    uses: string;
    confidence: string;
    processed_image_url: string;
}

// Define the type for chat messages
interface ChatMessage {
    role: 'user' | 'system';
    text: string;
}

// Define the main component
const IndexScreen = () => {
    const [image, setImage] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [result, setResult] = useState<PlantResult | null>(null);
    const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
    const [chatInput, setChatInput] = useState<string>('');
    
    // Reference for auto-scrolling the chat history
    const chatScrollViewRef = useRef<ScrollView>(null); 

    // 1. Request Permissions
    useEffect(() => {
        (async () => {
            if (Platform.OS !== 'web') {
                const { status: mediaStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
                if (mediaStatus !== 'granted') {
                    Alert.alert('Permission required', 'We need camera roll permissions to select images.');
                }
                await ImagePicker.requestCameraPermissionsAsync(); 
            }
        })();
    }, []);

    // 2. Auto-scroll Chat
    useEffect(() => {
        if (chatScrollViewRef.current) {
            chatScrollViewRef.current.scrollToEnd({ animated: true });
        }
    }, [chatHistory]);

    // 3. Image Picker Logic
    const pickImage = async (useCamera: boolean) => {
        let pickerResult;
        const options: ImagePicker.ImagePickerOptions = {
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.8,
        };

        if (useCamera) {
            pickerResult = await ImagePicker.launchCameraAsync(options);
        } else {
            pickerResult = await ImagePicker.launchImageLibraryAsync(options);
        }

        if (!pickerResult.canceled) {
            const uri = pickerResult.assets[0].uri;
            setImage(uri);
            setResult(null); 
            setChatHistory([]); // Clear chat history for new image
            uploadImage(uri);
        }
    };

    // 4. Image Upload and Classification API Call
    const uploadImage = async (uri: string) => {
        setLoading(true);
        setResult(null);

        const formData = new FormData();
        const filename = uri.split('/').pop() || 'photo.jpg';
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : `image`;

        formData.append('file', {
            uri: uri,
            name: filename,
            type: type,
        } as any);

        try {
            const response = await axios.post(`${API_BASE_URL}/classify`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 60000, // 60 seconds
            });

            const data = response.data as PlantResult;
            setResult(data);
            
            // Generate initial system message
            let initialChatText = `🌱 Identification complete! The image has been classified as **${data.name}** with ${data.confidence} confidence.`;
            
            if (data.is_medicinal) {
                initialChatText += `\n\n**Status:** This is an Indian medicinal plant. Its primary uses are: ${data.uses}. Feel free to ask me more about its preparation, dosage, or properties.`;
            } else {
                initialChatText += `\n\n**Status:** This plant is currently classified as non-medicinal in our domain-specific database. You can still ask for general information.`;
            }

            setChatHistory([{ role: 'system', text: initialChatText }]);

        } catch (error) {
            console.error('Classification Error:', error);
            const errorMessage = axios.isAxiosError(error) && error.response 
                ? error.response.data.error 
                : error instanceof Error ? error.message : 'Unknown error';
            Alert.alert('Upload Failed', `Could not connect to ${API_BASE_URL}. Ensure your laptop and phone are on the same WiFi. Error: ${errorMessage}`);
        } finally {
            setLoading(false);
        }
    };

    // 5. Chat API Call
    const sendQuery = async () => {
        if (!chatInput.trim() || !result || loading) return; 

        const userQuery = chatInput;
        setChatInput('');

        const newHistory: ChatMessage[] = [...chatHistory, { role: 'user', text: userQuery }];
        setChatHistory(newHistory);

        try {
            const response = await axios.post(`${API_BASE_URL}/chat`, {
                query: userQuery,
                plant_name: result.name, 
            });

            setChatHistory(prev => [...prev, { role: 'system', text: response.data.response }]);
        } catch (error) {
            console.error('Chat Error:', (error as Error).message);
            setChatHistory(prev => [...prev, { role: 'system', text: 'Sorry, I failed to connect to the AI bot.' }]);
        }
    };


    const renderHeader = () => (
        <View style={styles.header}>
            <Text style={styles.headerTitle}>🌿 Indian Medicinal Plant AI</Text>
        </View>
    );

    const renderResultCard = () => {
        if (!result) return null;
        return (
            <View style={styles.card}>
                <Text style={styles.cardTitle}>🔬 Classification Result</Text>
                
                <View style={styles.infoRow}>
                    <Text style={styles.resultText}>Plant Name:</Text> 
                    <Text style={styles.resultValue}>{result.name}</Text>
                </View>
                
                <View style={styles.infoRow}>
                    <Text style={styles.resultText}>Confidence:</Text>
                    <Text style={styles.resultValue}>{result.confidence}</Text>
                </View>

                <View style={styles.infoRow}>
                    <Text style={styles.resultText}>Status:</Text>
                    <Text style={result.is_medicinal ? styles.medicinal : styles.notMedicinal}>
                        {result.is_medicinal ? 'MEDICINAL PLANT' : 'NON-MEDICINAL'}
                    </Text>
                </View>
                
                {result.is_medicinal && (
                    <View style={styles.usesSection}>
                        <Text style={styles.resultTextBold}>Traditional Uses:</Text>
                        <Text style={styles.usesText}>{result.uses}</Text>
                    </View>
                )}

                {/* Display processed image (Base64) */}
                {result.processed_image_url && (
                    <View style={styles.imageDisplaySection}>
                        <Text style={styles.resultTextBold}>Processed Image (Cleaned):</Text>
                        <Image 
                            source={{ uri: result.processed_image_url }} 
                            style={styles.processedImage} 
                            resizeMode="contain" 
                        />
                    </View>
                )}
            </View>
        );
    };

    const renderChatInterface = () => {
        if (!result) return null;

        return (
            <View style={styles.chatContainer}>
                <Text style={styles.chatTitle}>🤖 Talk to the AI Assistant</Text>
                
                {/* FIX APPLIED HERE: nestedScrollEnabled and fixed height */}
                <ScrollView 
                    style={styles.chatHistory} 
                    ref={chatScrollViewRef}
                    contentContainerStyle={styles.chatHistoryContent}
                    nestedScrollEnabled={true} 
                    showsVerticalScrollIndicator={true}
                >
                    {chatHistory.map((message, index) => (
                        <View key={index} style={[
                            styles.messageBubble,
                            message.role === 'user' ? styles.userBubble : styles.systemBubble
                        ]}>
                            <FontAwesome 
                                name={message.role === 'user' ? "user-circle" : "android"} 
                                size={18} 
                                color={message.role === 'user' ? "#4F46E5" : "#10B981"} 
                                style={{ marginRight: 8 }}
                            />
                            <Text style={message.role === 'user' ? styles.userText : styles.systemText}>
                                {message.text}
                            </Text>
                        </View>
                    ))}
                </ScrollView>

                <View style={styles.chatInputContainer}>
                    <TextInput
                        style={styles.input}
                        placeholder={`Ask about ${result.name}...`}
                        placeholderTextColor="#9ca3af"
                        value={chatInput}
                        onChangeText={setChatInput}
                        onSubmitEditing={sendQuery} // Optional
                        editable={!loading}
                        multiline={true} 
                        textAlignVertical="top" 
                    />
                    <TouchableOpacity 
                        style={styles.sendButton} 
                        onPress={sendQuery}
                        disabled={!chatInput.trim() || loading} 
                    >
                        {loading ? 
                            <ActivityIndicator size="small" color="#fff" /> : 
                            <MaterialIcons name="send" size={24} color="#fff" />
                        }
                    </TouchableOpacity>
                </View>
            </View>
        );
    };

    return (
        <KeyboardAvoidingView
            style={styles.container}
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            keyboardVerticalOffset={Platform.OS === 'ios' ? 40 : 0} 
        >
            {renderHeader()}

            {/* FIX APPLIED HERE: keyboardShouldPersistTaps */}
            <ScrollView 
                contentContainerStyle={styles.content}
                keyboardShouldPersistTaps="handled"
            >
                
                {/* Image Preview / Placeholder */}
                {image ? (
                    <View style={styles.imagePreviewContainer}>
                        <Image source={{ uri: image }} style={styles.imagePreview} />
                        {loading && (
                            <View style={styles.loadingOverlay}>
                                <ActivityIndicator size="large" color="#4F46E5" />
                                <Text style={styles.loadingText}>Processing...</Text>
                            </View>
                        )}
                    </View>
                ) : (
                    <View style={styles.placeholder}>
                        <MaterialIcons name="image" size={width * 0.2} color="#94a3b8" />
                        <Text style={styles.placeholderText}>Upload or Capture a Plant Image</Text>
                    </View>
                )}

                {/* Action Buttons */}
                <View style={styles.buttonRow}>
                    <TouchableOpacity style={styles.button} onPress={() => pickImage(false)} disabled={loading}>
                        <MaterialIcons name="folder-open" size={24} color="#fff" />
                        <Text style={styles.buttonText}>Upload Image</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.button} onPress={() => pickImage(true)} disabled={loading}>
                        <MaterialIcons name="camera-alt" size={24} color="#fff" />
                        <Text style={styles.buttonText}>Capture Photo</Text>
                    </TouchableOpacity>
                </View>

                {renderResultCard()}
                {renderChatInterface()}

            </ScrollView>
        </KeyboardAvoidingView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f8fafc',
    },
    header: {
        paddingTop: 50,
        paddingBottom: 20,
        paddingHorizontal: 20,
        backgroundColor: '#4F46E5',
        borderBottomLeftRadius: 20,
        borderBottomRightRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.2,
        shadowRadius: 6,
        elevation: 8,
    },
    headerTitle: {
        fontSize: 24,
        fontWeight: '800',
        color: '#fff',
        textAlign: 'center',
    },
    content: {
        padding: 20,
        alignItems: 'center',
        paddingBottom: 100,
    },
    imagePreviewContainer: {
        width: '100%',
        maxHeight: height * 0.4, 
        aspectRatio: 1.33,
        borderRadius: 15,
        overflow: 'hidden',
        marginBottom: 20,
        position: 'relative',
        backgroundColor: '#e2e8f0',
        borderWidth: 1,
        borderColor: '#e5e7eb',
    },
    imagePreview: {
        width: '100%',
        height: '100%',
    },
    loadingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    loadingText: {
        marginTop: 10,
        fontSize: 14,
        fontWeight: '600',
        color: '#4F46E5',
    },
    placeholder: {
        width: '100%',
        aspectRatio: 1.33,
        backgroundColor: '#f1f5f9',
        borderRadius: 15,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
        borderWidth: 3,
        borderColor: '#e2e8f0',
        borderStyle: 'dashed',
    },
    placeholderText: {
        marginTop: 15,
        fontSize: 16,
        color: '#94a3b8',
        fontWeight: '600',
    },
    buttonRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '100%',
        marginBottom: 20,
    },
    button: {
        flexDirection: 'row',
        backgroundColor: '#10B981',
        padding: 15,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        width: '48%',
        elevation: 6,
    },
    buttonText: {
        marginLeft: 8,
        color: '#fff',
        fontSize: 16,
        fontWeight: '700',
    },
    card: {
        backgroundColor: '#fff',
        borderRadius: 15,
        padding: 20,
        width: '100%',
        marginBottom: 20,
        elevation: 4,
        borderLeftWidth: 6,
        borderColor: '#4F46E5',
    },
    cardTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#374151',
        marginBottom: 15,
        borderBottomWidth: 1,
        borderBottomColor: '#f3f4f6',
        paddingBottom: 8,
    },
    infoRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8,
    },
    resultText: {
        fontSize: 15,
        color: '#4b5563',
        fontWeight: '500',
    },
    resultTextBold: {
        fontSize: 15,
        color: '#1f2937',
        fontWeight: '600',
        marginBottom: 5,
    },
    resultValue: {
        fontWeight: '700',
        color: '#1f2937',
        maxWidth: '60%',
        textAlign: 'right',
    },
    medicinal: {
        fontWeight: '800',
        color: '#10B981',
    },
    notMedicinal: {
        fontWeight: '800',
        color: '#EF4444',
    },
    usesSection: {
        marginTop: 15,
        paddingTop: 10,
        borderTopWidth: 1,
        borderTopColor: '#f3f4f6',
    },
    usesText: {
        fontStyle: 'italic',
        color: '#6b7280',
        lineHeight: 20,
        fontSize: 14,
    },
    imageDisplaySection: {
        marginTop: 15,
        paddingTop: 10,
        borderTopWidth: 1,
        borderTopColor: '#f3f4f6',
    },
    processedImage: {
        width: '100%',
        height: 180,
        borderRadius: 10,
        marginTop: 10,
        backgroundColor: '#F3F4F6',
        borderWidth: 1,
        borderColor: '#e5e7eb',
    },
    
    // --- Chat styles ---
    chatContainer: {
        width: '100%',
        backgroundColor: '#fff',
        borderRadius: 15,
        padding: 15,
        elevation: 4,
    },
    chatTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#374151',
        marginBottom: 10,
        borderBottomWidth: 1,
        borderBottomColor: '#f3f4f6',
        paddingBottom: 5,
    },
    chatHistory: {
        // FIX: Using fixed height for nested scrolling
        height: 250, 
        marginBottom: 15,
        borderWidth: 1,
        borderColor: '#e5e7eb',
        borderRadius: 10,
        backgroundColor: '#f9fafb',
    },
    chatHistoryContent: {
        paddingHorizontal: 8,
        paddingVertical: 10,
        paddingBottom: 20,
    },
    messageBubble: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        padding: 12,
        borderRadius: 15,
        maxWidth: '90%',
        marginBottom: 10,
        elevation: 1,
    },
    userBubble: {
        backgroundColor: '#e0e7ff',
        alignSelf: 'flex-end',
        borderBottomRightRadius: 5,
    },
    systemBubble: {
        backgroundColor: '#ecfdf5',
        alignSelf: 'flex-start',
        borderBottomLeftRadius: 5,
    },
    userText: {
        color: '#1f2937',
        flexShrink: 1,
    },
    systemText: {
        color: '#065f46',
        flexShrink: 1,
    },
    chatInputContainer: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        borderTopWidth: 1,
        borderTopColor: '#e5e7eb',
        paddingTop: 10,
    },
    input: {
        flex: 1,
        backgroundColor: '#f9fafb',
        borderRadius: 10, 
        paddingHorizontal: 15,
        paddingVertical: 10,
        marginRight: 10,
        borderWidth: 1,
        borderColor: '#d1d5db',
        fontSize: 16,
        minHeight: 48,
        maxHeight: 120,
    },
    sendButton: {
        backgroundColor: '#4F46E5',
        width: 48,
        height: 48,
        borderRadius: 24,
        justifyContent: 'center',
        alignItems: 'center',
    },
});

export default IndexScreen;