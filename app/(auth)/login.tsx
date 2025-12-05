// app/(auth)/login.tsx - Dark Fintech Login (Demo)
import { Link, useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

export default function LoginScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter();

  const handleLogin = () => {
    if (username === 'user' && password === 'user123') {
      // Success! Go to main app
      router.replace('/(tabs)/predict/predict');   // ← this takes you to your home tabs
    } else {
      Alert.alert('Login Failed', 'Wrong username or password');
    }
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
        <Text style={styles.title}>Welcome Back</Text>
        <Text style={styles.subtitle}>Sign in to your Playground account</Text>

        {/* Username Input */}
        <TextInput
          style={styles.input}
          placeholder="Username"
          placeholderTextColor="#666"
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
        />

        {/* Password Input */}
        <TextInput
          style={styles.input}
          placeholder="Password"
          placeholderTextColor="#666"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          autoCapitalize="none"
        />

        {/* Demo Credentials Hint */}
        <Text style={styles.hint}>
          Demo: user / user123
        </Text>

        {/* Login Button */}
        <Pressable style={styles.loginButton} onPress={handleLogin}>
          <Text style={styles.loginButtonText}>Sign In</Text>
        </Pressable>

        {/* Forgot Password */}
        <Pressable>
          <Text style={styles.forgot}>Forgot Password?</Text>
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
  hint: {
    color: '#00d4ff',
    textAlign: 'center',
    marginBottom: 30,
    fontSize: 14,
    fontWeight: '600',
  },
  loginButton: {
    backgroundColor: '#00d4ff',
    paddingVertical: 18,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 20,
  },
  loginButtonText: {
    color: '#000',
    fontSize: 18,
    fontWeight: 'bold',
  },
  forgot: {
    color: '#888',
    textAlign: 'center',
    fontSize: 14,
  },
});