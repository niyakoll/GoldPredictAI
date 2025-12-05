// app/welcome.tsx - Dark Modern Fintech Welcome Screen
import { LinearGradient } from 'expo-linear-gradient';
import { Link } from 'expo-router';
import { ImageBackground, Pressable, StyleSheet, Text, View } from 'react-native';
export default function WelcomeScreen() {
  return (
    <ImageBackground
        source={require('../assets/images/stock_market.jpg')}   // ← this line changed
        style={styles.background}
        blurRadius={8}
    >
      <LinearGradient
        //colors={['rgba(248, 241, 241, 0.46)', 'rgba(255, 255, 255, 0.78)']}
        colors={['rgba(0,0,0,0.8)', 'rgba(0,0,0,0.95)']}
        style={StyleSheet.absoluteFillObject}
      />

      <View style={styles.container}>
        {/* App Logo / Icon */}
        <View style={styles.logoContainer}>
          <Text style={styles.logo}>📈</Text>
        </View>

        {/* Main Title */}
        <Text style={styles.subtitle}>Welcome to </Text>
        <Text style={styles.title}>Market Predict Playground!</Text>
        <Text style={styles.subtitle}>
          Leverage AI Power{'\n'}
          Build Your Furture.
        </Text>

        {/* Feature Highlights */}
        <View style={styles.features}>
          <Text style={styles.feature}>✓ Predict Gold Price with AI</Text>
          <Text style={styles.feature}>✓ Customise Setting for testing</Text>
          <Text style={styles.feature}>✓ Contribute Your setting to community</Text>
        </View>

        {/* Action Buttons */}
        <View style={styles.buttonContainer}>
          <Link href="/(auth)/login" asChild>
            <Pressable style={styles.primaryButton}>
              <Text style={styles.primaryButtonText}>Sign In</Text>
            </Pressable>
          </Link>

          <Link href="/(auth)/register" asChild>
            <Pressable style={styles.secondaryButton}>
              <Text style={styles.secondaryButtonText}>Create Account</Text>
            </Pressable>
          </Link>
        </View>
    
        {/* Footer */}
        <Text style={styles.footer}>
          By continuing, you agree to our Terms & Privacy Policy
        </Text>
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  container: {
    flex: 1,
    justifyContent: 'flex-end',
    paddingHorizontal: 32,
    paddingBottom: 60,
  },
  logoContainer: {
    alignSelf: 'center',
    marginBottom: 60,
    marginTop: 100,
  },
  logo: {
    fontSize: 80,
  },
  title: {
    fontSize: 42,
    fontWeight: '800',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 16,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 18,
    color: '#aaa',
    textAlign: 'center',
    lineHeight: 26,
    marginBottom: 40,
  },
  features: {
    marginBottom: 50,
    gap: 12,
  },
  feature: {
    color: '#4ade80',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonContainer: {
    gap: 16,
    marginBottom: 30,
  },
  primaryButton: {
    backgroundColor: '#00d4ff',
    paddingVertical: 18,
    borderRadius: 16,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#000',
    fontSize: 18,
    fontWeight: 'bold',
  },
  secondaryButton: {
    borderColor: '#444',
    borderWidth: 2,
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  footer: {
    color: '#666',
    fontSize: 12,
    textAlign: 'center',
  },
});