// app/(tabs)/predict/gold.tsx
import DateTimePicker from '@react-native-community/datetimepicker';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';

export default function GoldPredictScreen() {
  const router = useRouter();
  const [date, setDate] = useState(new Date());
  const [aiMode, setAiMode] = useState(2);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);

  const runPrediction = () => {
    const result = (Math.random() * 300 + 4000).toFixed(4);
    setPrediction(`$${result}`);
    Alert.alert('Prediction Ready!', `Gold: $${result} on ${date.toDateString()}`);
  };

  return (
    <View style={styles.container}>
      <Pressable style={styles.back} onPress={() => router.back()}>
        <Text style={styles.backText}>← Back</Text>
      </Pressable>

      <Text style={styles.title}>Gold Price Predict</Text>

      <Pressable style={styles.setting} onPress={() => setShowDatePicker(true)}>
        <Text style={styles.label}>Target Date</Text>
        <Text style={styles.value}>{date.toDateString()}</Text>
      </Pressable>

      {showDatePicker && (
        <DateTimePicker
          value={date}
          mode="date"
          minimumDate={new Date()}
          onChange={(e, d) => {
            setShowDatePicker(false);
            if (d) setDate(d);
          }}
        />
      )}

      <View style={styles.modes}>
        {[1, 2, 3].map(m => (
          <Pressable
            key={m}
            style={[styles.mode, aiMode === m && styles.active]}
            onPress={() => setAiMode(m)}
          >
            <Text style={aiMode === m ? styles.activeText : styles.modeText}>
              Mode {m}
            </Text>
          </Pressable>
        ))}
      </View>

      <Pressable style={styles.predict} onPress={runPrediction}>
        <Text style={styles.predictText}>Run AI Prediction</Text>
      </Pressable>

      {prediction && (
        <Text style={styles.result}>Predicted: {prediction}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f1e', padding: 20 },
  back: { position: 'absolute', top: 50, left: 20, zIndex: 10 },
  backText: { fontSize: 36, color: '#fff' },
  title: { fontSize: 32, fontWeight: '800', color: '#fff', textAlign: 'center', marginTop: 80 },
  setting: { backgroundColor: '#1a1a2e', padding: 20, borderRadius: 16, marginVertical: 10 },
  label: { color: '#00d4ff', fontWeight: '600' },
  value: { color: '#fff', fontSize: 18, marginTop: 8 },
  modes: { flexDirection: 'row', justifyContent: 'center', gap: 16, marginVertical: 30 },
  mode: { paddingHorizontal: 24, paddingVertical: 12, borderRadius: 12, backgroundColor: '#333' },
  active: { backgroundColor: '#00d4ff' },
  modeText: { color: '#fff' },
  activeText: { color: '#000', fontWeight: 'bold' },
  predict: { backgroundColor: '#00d4ff', padding: 20, borderRadius: 16, alignItems: 'center' },
  predictText: { color: '#000', fontSize: 18, fontWeight: 'bold' },
  result: { fontSize: 32, color: '#4ade80', textAlign: 'center', marginTop: 40, fontWeight: 'bold' },
});