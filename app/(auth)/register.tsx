// app/(auth)/register.tsx - Dark Fintech Register (Demo)
import { Link, useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

export default function RegisterScreen() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter();

  const handleRegister = () => {
    // Simple fake validation (just for demo feel)
    if (!name || !email || !password) {
      Alert.alert('Error', 'Please fill all fields');
      return;
    }
    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters');
      return;
    }

    // Fake success → go to login
    Alert.alert(
      'Success! 🎉',
      'Account created (demo only)',
      [{ text: 'OK', onPress: () => router.replace('/(auth)/login') }]
    );
  };

  return (
    <View style={styles.container}>
      {/* Back button */}
      <Link href="/welcome" asChild>
        <Pressable style={styles.backButton}>
          <Text style={styles.backText}>←</Text>
        </Pressable>
      </Link>

      <View style={styles.content}>
        <Text style={styles.title}>Create Account</Text>
        <Text style={styles.subtitle}>Join Playground today</Text>

        {/* Name Input */}
        <TextInput
          style={styles.input}
          placeholder="Full Name"
          placeholderTextColor="#666"
          value={name}
          onChangeText={setName}
        />

        {/* Email Input */}
        <TextInput
          style={styles.input}
          placeholder="Email Address"
          placeholderTextColor="#666"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />

        {/* Password Input */}
        <TextInput
          style={styles.input}
          placeholder="Password (6+ characters)"
          placeholderTextColor="#666"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          autoCapitalize="none"
        />

        {/* Register Button */}
        <Pressable style={styles.registerButton} onPress={handleRegister}>
          <Text style={styles.registerButtonText}>Create Account</Text>
        </Pressable>

        {/* Already have account? */}
        <Pressable onPress={() => router.push('/(auth)/login')}>
          <Text style={styles.loginLink}>
            Already have an account? <Text style={styles.bold}>Sign In</Text>
          </Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f1e',
  },
  backButton: {
    position: 'absolute',
    top: 60,
    left: 20,
    zIndex: 10,
  },
  backText: {
    fontSize: 32,
    color: '#fff',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  title: {
    fontSize: 36,
    fontWeight: '800',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
    marginBottom: 50,
  },
  input: {
    backgroundColor: '#1a1a2e',
    color: '#fff',
    paddingHorizontal: 20,
    paddingVertical: 18,
    borderRadius: 16,
    fontSize: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#333',
  },
  registerButton: {
    backgroundColor: '#00d4ff',
    paddingVertical: 18,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 20,
  },
  registerButtonText: {
    color: '#000',
    fontSize: 18,
    fontWeight: 'bold',
  },
  loginLink: {
    color: '#888',
    textAlign: 'center',
    fontSize: 14,
  },
  bold: {
    color: '#00d4ff',
    fontWeight: 'bold',
  },
});