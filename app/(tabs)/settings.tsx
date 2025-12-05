import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';

export default function SettingsScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings</Text>
      
      <Pressable style={styles.logout} onPress={() => router.replace('/welcome')}>
        <Text style={styles.logoutText}>Logout</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f1e', padding: 20 },
  title: { fontSize: 32, fontWeight: '800', color: '#fff', marginTop: 20 },
  logout: {
    backgroundColor: '#ff4444',
    padding: 18,
    borderRadius: 16,
    alignItems: 'center',
    marginTop: 40,
  },
  logoutText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
});